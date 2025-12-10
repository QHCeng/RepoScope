"""

Core features:
- Fetch repo structure from GitHub / GitLab / Bitbucket / local
- Use LLM to determine wiki structure (XML)
- Use LLM to generate Markdown pages
- Cache wiki result to disk
- Export as markdown or JSON

"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import os
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Literal

import requests
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field

# -------------------------------
# Data Models
# -------------------------------

Importance = Literal["high", "medium", "low"]
RepoType = Literal["github", "gitlab", "bitbucket", "local"]


class WikiPage(BaseModel):
    """
    Model for a wiki page.
    """
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str # Should ideally be Literal['high', 'medium', 'low']
    relatedPages: List[str]

class ProcessedProjectEntry(BaseModel):
    id: str  # Filename
    owner: str
    repo: str
    name: str  # owner/repo
    repo_type: str # Renamed from type to repo_type for clarity with existing models
    submittedAt: int # Timestamp
    language: str # Extracted from filename

class RepoInfo(BaseModel):
    owner: str
    repo: str
    type: str
    token: Optional[str] = None
    localPath: Optional[str] = None
    repoUrl: Optional[str] = None


class WikiSection(BaseModel):
    """
    Model for the wiki sections.
    """
    id: str
    title: str
    pages: List[str]
    subsections: Optional[List[str]] = None


class WikiStructureModel(BaseModel):
    """
    Model for the overall wiki structure.
    """
    id: str
    title: str
    description: str
    pages: List[WikiPage]
    sections: Optional[List[WikiSection]] = None
    root_sections: Optional[List[str]] = None

class WikiCacheData(BaseModel):
    """
    Model for the data to be stored in the wiki cache.
    """
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None  #compatible for old cache
    repo: Optional[RepoInfo] = None
    provider: Optional[str] = None
    model: Optional[str] = None

class WikiCacheRequest(BaseModel):
    """
    Model for the request body when saving wiki cache.
    """
    repo: RepoInfo
    language: str
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    provider: str
    model: str

class WikiExportRequest(BaseModel):
    """
    Model for requesting a wiki export.
    """
    repo_url: str = Field(..., description="URL of the repository")
    pages: List[WikiPage] = Field(..., description="List of wiki pages to export")
    format: Literal["markdown", "json"] = Field(..., description="Export format (markdown or json)")

# --- Model Configuration Models ---
class Model(BaseModel):
    """
    Model for LLM model configuration
    """
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name for the model")

class Provider(BaseModel):
    """
    Model for LLM provider configuration
    """
    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Display name for the provider")
    models: List[Model] = Field(..., description="List of available models for this provider")
    supportsCustomModel: Optional[bool] = Field(False, description="Whether this provider supports custom models")

class ModelConfig(BaseModel):
    """
    Model for the entire model configuration
    """
    providers: List[Provider] = Field(..., description="List of available model providers")
    defaultProvider: str = Field(..., description="ID of the default provider")

class AuthorizationConfig(BaseModel):
    code: str = Field(..., description="Authorization code")

# -------------------------------
# Repo Fetcher (GitHub / GitLab / Bitbucket / local)
# -------------------------------

class RepoFetcher:
    def __init__(self, repo: RepoInfo):
        self.repo = repo

    # ---------- Headers ----------
    def _github_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.repo.token:
            headers["Authorization"] = f"Bearer {self.repo.token}"
        return headers

    def _gitlab_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.repo.token:
            headers["PRIVATE-TOKEN"] = self.repo.token
        return headers

    def _bitbucket_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.repo.token:
            headers["Authorization"] = f"Bearer {self.repo.token}"
        return headers

    # ---------- Public API ----------
    def fetch_structure_and_readme(self) -> tuple[str, str]:
        """
        Returns:
            file_tree (str): one path per line
            readme (str)
        """
        if self.repo.type == "github":
            return self._fetch_github()
        elif self.repo.type == "gitlab":
            return self._fetch_gitlab()
        elif self.repo.type == "bitbucket":
            return self._fetch_bitbucket()
        elif self.repo.type == "local":
            return self._fetch_local()
        else:
            raise ValueError(f"Unsupported repo type: {self.repo.type}")

    # ---------- GitHub ----------
    def _github_api_base(self) -> str:
        # For GitHub Enterprise you could derive from repo.repo_url
        return "https://api.github.com"

    def _fetch_github(self) -> tuple[str, str]:
        base = self._github_api_base()
        # 1. Get repo info to find default branch
        info_url = f"{base}/repos/{self.repo.owner}/{self.repo.repo}"
        resp = requests.get(info_url, headers=self._github_headers())
        resp.raise_for_status()
        info = resp.json()
        default_branch = info.get("default_branch", "main")
        self.repo.default_branch = default_branch

        # 2. Get tree
        tree_url = f"{base}/repos/{self.repo.owner}/{self.repo.repo}/git/trees/{default_branch}?recursive=1"
        resp = requests.get(tree_url, headers=self._github_headers())
        resp.raise_for_status()
        data = resp.json()
        tree = data.get("tree", [])

        file_tree = "\n".join(
            item["path"] for item in tree if item.get("type") == "blob"
        )

        # 3. Readme
        readme = ""
        readme_url = f"{base}/repos/{self.repo.owner}/{self.repo.repo}/readme"
        r_rm = requests.get(readme_url, headers=self._github_headers())
        if r_rm.ok:
            rm_json = r_rm.json()
            content_b64 = rm_json.get("content", "")
            try:
                readme = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
            except Exception:
                readme = ""
        return file_tree, readme

    # ---------- GitLab ----------
    def _fetch_gitlab(self) -> tuple[str, str]:
        if not self.repo.repo_url:
            raise ValueError("GitLab repo_url is required")

        # Extract GitLab base url and project path
        from urllib.parse import urlparse

        u = urlparse(self.repo.repo_url)
        base = f"{u.scheme}://{u.netloc}"
        project_path = u.path.lstrip("/").rstrip(".git")
        project_id = requests.utils.quote(project_path, safe="")

        # 1. Get project info
        project_url = f"{base}/api/v4/projects/{project_id}"
        resp = requests.get(project_url, headers=self._gitlab_headers())
        resp.raise_for_status()
        info = resp.json()
        default_branch = info.get("default_branch", "main")
        self.repo.default_branch = default_branch

        # 2. List repository tree (paginated)
        page = 1
        file_entries = []
        while True:
            tree_url = (
                f"{project_url}/repository/tree"
                f"?recursive=true&per_page=100&page={page}"
            )
            r = requests.get(tree_url, headers=self._gitlab_headers())
            r.raise_for_status()
            page_data = r.json()
            if not page_data:
                break
            file_entries.extend(page_data)

            next_page = r.headers.get("X-Next-Page")
            if not next_page:
                break
            page = int(next_page)

        file_tree = "\n".join(
            item["path"] for item in file_entries if item.get("type") == "blob"
        )

        # 3. Readme
        readme = ""
        readme_url = f"{project_url}/repository/files/README.md/raw?ref={default_branch}"
        r_rm = requests.get(readme_url, headers=self._gitlab_headers())
        if r_rm.ok:
            readme = r_rm.text
        return file_tree, readme

    # ---------- Bitbucket ----------
    def _fetch_bitbucket(self) -> tuple[str, str]:
        if not self.repo.repo_url:
            raise ValueError("Bitbucket repo_url is required")

        from urllib.parse import urlparse

        u = urlparse(self.repo.repo_url)
        base = f"{u.scheme}://{u.netloc}"
        repo_path = u.path.lstrip("/")

        # 1. Get repo info to get default branch
        info_url = f"{base}/2.0/repositories/{repo_path}"
        resp = requests.get(info_url, headers=self._bitbucket_headers())
        resp.raise_for_status()
        info = resp.json()
        main_branch = info.get("mainbranch", {}).get("name", "main")
        self.repo.default_branch = main_branch

        # 2. Get file listing
        src_url = f"{base}/2.0/repositories/{repo_path}/src/{main_branch}/?recursive=true&pagelen=100"
        resp = requests.get(src_url, headers=self._bitbucket_headers())
        resp.raise_for_status()
        data = resp.json()
        values = data.get("values", [])

        file_tree = "\n".join(
            item["path"] for item in values if item.get("type") == "commit_file"
        )

        # 3. Readme
        readme = ""
        readme_url = f"{base}/2.0/repositories/{repo_path}/src/{main_branch}/README.md"
        r_rm = requests.get(readme_url, headers=self._bitbucket_headers())
        if r_rm.ok:
            readme = r_rm.text
        return file_tree, readme

    # ---------- Local ----------
    def _fetch_local(self) -> tuple[str, str]:
        if not self.repo.local_path:
            raise ValueError("local_path is required for local repo")

        root = Path(self.repo.local_path).expanduser().resolve()
        paths = []
        for p in root.rglob("*"):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                paths.append(rel)
        file_tree = "\n".join(paths)

        readme = ""
        readme_path = root / "README.md"
        if readme_path.exists():
            readme = readme_path.read_text(encoding="utf-8", errors="ignore")
        return file_tree, readme


# -------------------------------
# LLM Client
# -------------------------------

class LLMClient:
    """
    Minimal wrapper around an LLM endpoint.

    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or "https://api.openai.com/v1"

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
        }
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# -------------------------------
# Wiki Generator Service
# -------------------------------

class WikiService:
    def __init__(
        self,
        repo: RepoInfo,
        llm: LLMClient,
        language: str = "en",
        comprehensive: bool = True,
        cache_dir: str = ".deepwiki_cache",
    ):
        self.repo = repo
        self.llm = llm
        self.language = language
        self.comprehensive = comprehensive
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Cache ----------
    def _cache_key(self) -> Path:
        style = "comprehensive" if self.comprehensive else "concise"
        name = f"{self.repo.type}_{self.repo.owner}_{self.repo.repo}_{self.language}_{style}.json"
        return self.cache_dir / name

    def load_cache(self) -> Optional[tuple[WikiStructureModel, Dict[str, WikiPage]]]:
        p = self._cache_key()
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        pages = [
            WikiPage(
                id=p["id"],
                title=p["title"],
                file_paths=p["file_paths"],
                importance=p["importance"],
                related_pages=p.get("related_pages", []),
                content=p.get("content", ""),
            )
            for p in data["wiki_structure"]["pages"]
        ]
        sections = [
            WikiSection(
                id=s["id"],
                title=s["title"],
                pages=s["pages"],
                subsections=s.get("subsections", []),
            )
            for s in data["wiki_structure"].get("sections", [])
        ]
        wiki = WikiStructureModel(
            id=data["wiki_structure"]["id"],
            title=data["wiki_structure"]["title"],
            description=data["wiki_structure"]["description"],
            pages=pages,
            sections=sections,
            root_sections=data["wiki_structure"].get("root_sections", []),
        )
        page_dict = {p.id: p for p in pages}
        return wiki, page_dict

    def save_cache(self, wiki: WikiStructureModel):
        p = self._cache_key()
        serializable = {
            "repo": dataclasses.asdict(self.repo),
            "language": self.language,
            "comprehensive": self.comprehensive,
            "wiki_structure": {
                "id": wiki.id,
                "title": wiki.title,
                "description": wiki.description,
                "pages": [dataclasses.asdict(page) for page in wiki.pages],
                "sections": [dataclasses.asdict(sec) for sec in wiki.sections],
                "root_sections": wiki.root_sections,
            },
        }
        p.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- Wiki structure ----------
    def _language_label(self) -> str:
        mapping = {
            "en": "English",
            "ja": "Japanese (日本語)",
            "zh": "Mandarin Chinese (中文)",
            "zh-tw": "Traditional Chinese (繁體中文)",
            "es": "Spanish (Español)",
            "kr": "Korean (한국어)",
            "vi": "Vietnamese (Tiếng Việt)",
            "pt-br": "Brazilian Portuguese (Português Brasileiro)",
            "fr": "Français (French)",
            "ru": "Русский (Russian)",
        }
        return mapping.get(self.language, "English")

    def determine_wiki_structure(self, file_tree: str, readme: str) -> WikiStructureModel:
        """Call LLM to get <wiki_structure> XML and parse it."""
        style_text = "comprehensive" if self.comprehensive else "concise"
        num_pages = "8-12" if self.comprehensive else "4-6"

        prompt = textwrap.dedent(f"""
        Analyze this repository and create a wiki structure for it.

        1. The complete file tree:
        <file_tree>
        {file_tree}
        </file_tree>

        2. The README:
        <readme>
        {readme}
        </readme>

        The wiki content will be generated in {self._language_label()}.

        Create a {style_text} wiki with {num_pages} pages.

        Return ONLY XML in the following format:

        <wiki_structure>
          <title>[Overall title for the wiki]</title>
          <description>[Brief description of the repository]</description>
          <pages>
            <page id="page-1">
              <title>[Page title]</title>
              <description>[Brief description of what this page will cover]</description>
              <importance>high|medium|low</importance>
              <relevant_files>
                <file_path>[Path to a relevant file]</file_path>
              </relevant_files>
              <related_pages>
                <related>page-2</related>
              </related_pages>
            </page>
          </pages>
        </wiki_structure>

        IMPORTANT:
        - DO NOT wrap in markdown code fences.
        - Start with <wiki_structure> and end with </wiki_structure>.
        - relevant_files MUST use actual paths from the file_tree.
        """).strip()

        xml_text = self.llm.chat([{"role": "user", "content": prompt}])
        # Remove accidental code fences
        xml_text = xml_text.strip()
        if xml_text.startswith("```"):
            xml_text = xml_text.strip("`")
            xml_text = xml_text.split("\n", 1)[-1]
        # Extract <wiki_structure>...</wiki_structure>
        if "<wiki_structure" not in xml_text:
            raise RuntimeError("LLM response does not contain <wiki_structure>")
        start = xml_text.find("<wiki_structure")
        end = xml_text.rfind("</wiki_structure>") + len("</wiki_structure>")
        xml_text = xml_text[start:end]

        root = ET.fromstring(xml_text)

        title = root.findtext("title", default="")
        desc = root.findtext("description", default="")
        pages: List[WikiPage] = []

        pages_el = root.find("pages")
        if pages_el is not None:
            for p_el in pages_el.findall("page"):
                pid = p_el.attrib.get("id", f"page-{len(pages) + 1}")
                p_title = p_el.findtext("title", default="")
                importance = p_el.findtext("importance", default="medium")
                if importance not in ("high", "medium", "low"):
                    importance = "medium"
                file_paths = [
                    fp_el.text or ""
                    for fp_el in p_el.findall("./relevant_files/file_path")
                ]
                related = [
                    r_el.text or ""
                    for r_el in p_el.findall("./related_pages/related")
                ]

                pages.append(
                    WikiPage(
                        id=pid,
                        title=p_title,
                        file_paths=file_paths,
                        importance=importance,  # type: ignore
                        related_pages=related,
                    )
                )

        # sections/root_sections
        return WikiStructureModel(
            id="wiki",
            title=title,
            description=desc,
            pages=pages,
            sections=[],
            root_sections=[],
        )

    # ---------- Page content ----------
    def generate_page_content(self, page: WikiPage) -> str:
        def generate_file_url(path: str) -> str:
            if self.repo.type == "local" or not self.repo.repo_url:
                return path
            base = self.repo.repo_url.rstrip("/")
            if self.repo.type == "github":
                return f"{base}/blob/{self.repo.default_branch}/{path}"
            elif self.repo.type == "gitlab":
                return f"{base}/-/blob/{self.repo.default_branch}/{path}"
            elif self.repo.type == "bitbucket":
                return f"{base}/src/{self.repo.default_branch}/{path}"
            return path

        file_links = "\n".join(
            f"- [{p}]({generate_file_url(p)})" for p in page.file_paths
        ) or "- (no files provided)"

        lang_label = self._language_label()

        prompt = textwrap.dedent(f"""
        You are an expert technical writer and software architect.
        Generate a technical wiki page in Markdown about the topic:

        [WIKI_PAGE_TOPIC]
        {page.title}

        Use ONLY the following source files as ground truth:
        [RELEVANT_SOURCE_FILES]
        {file_links}

        The wiki MUST start with a <details> block listing all files used, then '# {page.title}'.

        Requirements:
        - Use Mermaid diagrams (graph TD / sequenceDiagram) when appropriate.
        - Use tables for API, configs, data models.
        - Ground all statements in the provided files (do not hallucinate external knowledge).
        - Language: {lang_label}.

        Return ONLY valid Markdown (no extra explanations).
        """).strip()

        content = self.llm.chat([{"role": "user", "content": prompt}])
        # Remove accidental markdown fences
        if content.strip().startswith("```"):
            content = content.strip()
            # remove first line ```xxx
            content = "\n".join(content.splitlines()[1:])
            if content.strip().endswith("```"):
                content = "\n".join(content.splitlines()[:-1])
        return content

    # ---------- High-level orchestration ----------
    def generate_wiki(self, use_cache: bool = True) -> WikiStructureModel:
        if use_cache:
            cached = self.load_cache()
            if cached:
                wiki, _ = cached
                print("[deepwiki] Loaded wiki from cache.")
                return wiki

        # 1. Fetch repo structure
        fetcher = RepoFetcher(self.repo)
        file_tree, readme = fetcher.fetch_structure_and_readme()

        # 2. Determine wiki structure
        print("[deepwiki] Calling LLM to determine wiki structure...")
        wiki = self.determine_wiki_structure(file_tree, readme)

        # 3. Generate content for each page
        for page in wiki.pages:
            print(f"[deepwiki] Generating content for page: {page.title}")
            page.content = self.generate_page_content(page)

        # 4. Save cache
        self.save_cache(wiki)
        print("[deepwiki] Wiki generated and cached.")
        return wiki

    # ---------- Export ----------
    def export_markdown(self, wiki: WikiStructureModel, out_path: str):
        """
        Export all pages into a single markdown file with headings.
        """
        lines = [f"# {wiki.title}", "", wiki.description, "", "---", ""]
        for page in wiki.pages:
            lines.append(f"# {page.title}")
            lines.append("")
            lines.append(page.content)
            lines.append("\n---\n")
        Path(out_path).write_text("\n".join(lines), encoding="utf-8")
        print(f"[deepwiki] Exported markdown to {out_path}")

    def export_json(self, wiki: WikiStructureModel, out_path: str):
        data = {
            "title": wiki.title,
            "description": wiki.description,
            "pages": [dataclasses.asdict(p) for p in wiki.pages],
        }
        Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[deepwiki] Exported JSON to {out_path}")





