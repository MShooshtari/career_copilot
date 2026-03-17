"""Unit tests for add_job agent: URL helpers, title from URL, Indeed/BambooHR, extraction with mocked HTTP/LLM."""

from __future__ import annotations

from unittest.mock import patch

from career_copilot.agents.add_job import (
    _build_indeed_embedded_url,
    _company_from_bamboohr_subdomain,
    _generic_title_from_url,
    _indeed_company_from_path,
    _indeed_title_from_query,
    _is_search_results_url,
    _title_from_url_path,
    extract_job_from_text,
    extract_job_from_url,
    title_from_url_path,
)

# --- title_from_url_path / _title_from_url_path ---


def test_title_from_url_path_slug() -> None:
    assert title_from_url_path("https://example.com/careers/Senior-Data-Scientist") == "Senior Data Scientist"


def test_title_from_url_path_viewjob_empty() -> None:
    assert _title_from_url_path("https://ca.indeed.com/viewjob?jk=abc") == ""


def test_title_from_url_path_careers_segment_skipped() -> None:
    assert _title_from_url_path("https://example.com/careers") == ""


def test_title_from_url_path_uuid_skipped() -> None:
    assert _title_from_url_path("https://rippling.com/jobs/e610fefb-7086-45de-a4a9-fd0c5e9fa4e4") == ""


# --- _indeed_title_from_query ---


def test_indeed_title_from_query_with_q() -> None:
    assert _indeed_title_from_query("https://ca.indeed.com/viewjob?jk=1&q=machine+learning+engineer") == "Machine Learning Engineer"


def test_indeed_title_from_query_without_q() -> None:
    assert _indeed_title_from_query("https://ca.indeed.com/viewjob?jk=1") == ""


def test_indeed_title_from_query_non_indeed() -> None:
    assert _indeed_title_from_query("https://linkedin.com/jobs/123") == ""


# --- _indeed_company_from_path ---


def test_indeed_company_from_path_cmp() -> None:
    assert _indeed_company_from_path("https://ca.indeed.com/cmp/Ind-Technology-Pty-Ltd/jobs?jk=1") == "Ind Technology Pty Ltd"


def test_indeed_company_from_path_no_cmp() -> None:
    assert _indeed_company_from_path("https://ca.indeed.com/viewjob?jk=1") == ""


# --- _generic_title_from_url ---


def test_generic_title_indeed_with_q() -> None:
    assert _generic_title_from_url("https://ca.indeed.com/viewjob?jk=1&q=data+engineer") == "Data Engineer"


def test_generic_title_indeed_cmp_path() -> None:
    assert _generic_title_from_url("https://ca.indeed.com/cmp/Acme-Corp/jobs?jk=1") == "Job at Acme Corp"


def test_generic_title_indeed_fallback() -> None:
    assert _generic_title_from_url("https://ca.indeed.com/viewjob?jk=1") == "Job from Indeed"


def test_generic_title_bamboohr_subdomain() -> None:
    assert _generic_title_from_url("https://tractionrec.bamboohr.com/careers/135") == "Job at Tractionrec"


def test_generic_title_bamboohr_no_subdomain() -> None:
    assert _generic_title_from_url("https://bamboohr.com/careers") == "Job from BambooHR"


def test_generic_title_linkedin() -> None:
    assert _generic_title_from_url("https://linkedin.com/jobs/view/123") == "Job from LinkedIn"


def test_generic_title_generic_link() -> None:
    assert _generic_title_from_url("https://example.com/job/123") == "Job from link"


# --- _company_from_bamboohr_subdomain ---


def test_company_from_bamboohr_subdomain() -> None:
    assert _company_from_bamboohr_subdomain("https://tractionrec.bamboohr.com/careers/135") == "Tractionrec"


def test_company_from_bamboohr_subdomain_with_hyphen() -> None:
    assert _company_from_bamboohr_subdomain("https://my-company.bamboohr.com/careers") == "My Company"


def test_company_from_bamboohr_non_bamboohr() -> None:
    assert _company_from_bamboohr_subdomain("https://example.com") == ""


# --- _build_indeed_embedded_url ---


def test_build_indeed_embedded_url_viewjob_with_jk() -> None:
    url = "https://ca.indeed.com/viewjob?jk=12932043a48059d8&q=ml"
    out = _build_indeed_embedded_url(url)
    assert out is not None
    assert "viewjob" in out
    assert "viewtype=embedded" in out
    assert "jk=12932043a48059d8" in out


def test_build_indeed_embedded_url_cmp_jobs_with_jk() -> None:
    url = "https://ca.indeed.com/cmp/Ind-Technology-Pty-Ltd/jobs?jk=12932043a48059d8&start=0"
    out = _build_indeed_embedded_url(url)
    assert out is not None
    assert out.endswith("/viewjob") or "/viewjob?" in out
    assert "viewtype=embedded" in out
    assert "jk=12932043a48059d8" in out


def test_build_indeed_embedded_url_no_jk() -> None:
    assert _build_indeed_embedded_url("https://ca.indeed.com/viewjob") is None


def test_build_indeed_embedded_url_non_indeed() -> None:
    assert _build_indeed_embedded_url("https://linkedin.com/jobs/123") is None


def test_build_indeed_embedded_url_already_embedded() -> None:
    url = "https://ca.indeed.com/viewjob?viewtype=embedded&jk=1"
    # Current impl returns URL even when viewtype=embedded is in input; that's OK for tests.
    out = _build_indeed_embedded_url(url)
    # Implementation excludes when "viewtype=embedded" in url
    assert out is None


# --- _is_search_results_url ---


def test_is_search_results_url_indeed_viewjob_false() -> None:
    assert _is_search_results_url("https://ca.indeed.com/viewjob?jk=1") is False


def test_is_search_results_url_indeed_jobs_true() -> None:
    assert _is_search_results_url("https://ca.indeed.com/jobs?q=python") is True


def test_is_search_results_url_indeed_cmp_with_jk_false() -> None:
    assert _is_search_results_url("https://ca.indeed.com/cmp/Acme/jobs?jk=1") is False


# --- extract_job_from_text (mocked LLM) ---


def test_extract_job_from_text_mocked_llm() -> None:
    with patch("career_copilot.agents.add_job._extract_with_llm") as mock_llm:
        mock_llm.return_value = {
            "title": "Software Engineer",
            "company": "Acme Inc",
            "location": "Remote",
            "salary_min": 100000,
            "salary_max": 150000,
            "description": "Build things.",
            "skills": ["Python", "AWS"],
            "url": None,
        }
        out = extract_job_from_text("We are hiring a Software Engineer at Acme Inc. Remote. $100k-150k.")
        assert out["title"] == "Software Engineer"
        assert out["company"] == "Acme Inc"
        assert out["location"] == "Remote"
        assert out["salary_min"] == 100000
        assert out["salary_max"] == 150000
        assert out["skills"] == ["Python", "AWS"]
        mock_llm.assert_called_once()


def test_extract_job_from_text_passes_hints() -> None:
    with patch("career_copilot.agents.add_job._extract_with_llm") as mock_llm:
        mock_llm.return_value = {}
        extract_job_from_text(
            "Job description here.",
            location="Toronto",
            salary_min=80,
            salary_max=120,
            url="https://example.com/job",
        )
        call_args = mock_llm.call_args
        assert call_args is not None
        assert "Toronto" in call_args[0][1] or "80" in call_args[0][1]
        assert "https://example.com/job" in call_args[0][1]


# --- extract_job_from_url (mocked fetch) ---


def test_extract_job_from_url_empty_url() -> None:
    assert extract_job_from_url("") == {}
    assert extract_job_from_url("   ") == {}


def test_extract_job_from_url_adds_https() -> None:
    with patch("career_copilot.agents.add_job._fetch_page_html", return_value="") as mock_fetch:
        with patch("career_copilot.agents.add_job._extract_with_llm") as mock_llm:
            mock_llm.return_value = {"title": "Job", "company": None, "location": None, "description": None, "skills": [], "url": None}
            extract_job_from_url("example.com/job")
            mock_fetch.assert_called_once()
            assert mock_fetch.call_args[0][0].startswith("https://")


def test_extract_job_from_url_indeed_embedded_html() -> None:
    html = """
    <script>
    "title":"Machine Learning Engineer"
    "companyName":"Acme Corp"
    "jobLocationCity":"Toronto"
    "description":"We need an ML engineer. Python, TensorFlow. Great benefits."
    </script>
    """
    with patch("career_copilot.agents.add_job._fetch_page_html", return_value=html):
        out = extract_job_from_url("https://ca.indeed.com/viewjob?jk=123")
    assert out.get("title") == "Machine Learning Engineer"
    assert out.get("company") == "Acme Corp"
    assert out.get("location") == "Toronto"
    assert out.get("description")
    assert out.get("url") == "https://ca.indeed.com/viewjob?jk=123"


def test_extract_job_from_url_jsonld_job_posting() -> None:
    html = """
    <script type="application/ld+json">
    {"@type": "JobPosting", "title": "Data Scientist", "hiringOrganization": {"name": "DataCo"},
     "jobLocation": "Remote", "description": "Analyze data."}
    </script>
    """
    with patch("career_copilot.agents.add_job._fetch_page_html", return_value=html):
        out = extract_job_from_url("https://example.com/job/1")
    assert out.get("title") == "Data Scientist"
    assert out.get("company") == "DataCo"
    assert out.get("location") == "Remote"
    assert out.get("description") == "Analyze data."


def test_extract_job_from_url_fallback_generic_title() -> None:
    with patch("career_copilot.agents.add_job._fetch_page_html", return_value="<html><body>Loading...</body></html>"):
        with patch("career_copilot.agents.add_job._extract_with_llm") as mock_llm:
            mock_llm.return_value = {"title": None, "company": None, "location": None, "description": None, "skills": [], "url": None}
            out = extract_job_from_url("https://ca.indeed.com/viewjob?jk=1&q=machine+learning")
    assert out.get("title") == "Machine Learning"

