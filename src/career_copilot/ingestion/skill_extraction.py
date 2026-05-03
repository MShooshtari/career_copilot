"""Deterministic skill tag extraction for job descriptions.

The extractor intentionally uses a curated taxonomy instead of broad keyword
matching so generic words like "software" or "engineering" do not become tags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class SkillRule:
    canonical: str
    aliases: tuple[str, ...]


SKILL_RULES: tuple[SkillRule, ...] = (
    SkillRule("A/B Testing", ("A/B testing", "AB testing", "split testing", "experiment design")),
    SkillRule("Acceptance Testing", ("acceptance testing", "UAT", "user acceptance testing")),
    SkillRule("Agile", ("Agile", "Scrum", "Kanban")),
    SkillRule("Airflow", ("Airflow", "Apache Airflow")),
    SkillRule("Amazon Redshift", ("Redshift", "Amazon Redshift")),
    SkillRule("Ansible", ("Ansible",)),
    SkillRule("Apache Kafka", ("Kafka", "Apache Kafka")),
    SkillRule("Apache Spark", ("Spark", "Apache Spark", "PySpark")),
    SkillRule("API Design", ("API design", "REST API", "REST APIs", "GraphQL API")),
    SkillRule("ASP.NET", ("ASP.NET", "ASP NET")),
    SkillRule("AWS", ("AWS", "Amazon Web Services")),
    SkillRule("Azure", ("Azure", "Microsoft Azure")),
    SkillRule("BigQuery", ("BigQuery", "Google BigQuery")),
    SkillRule("CI/CD", ("CI/CD", "CI CD", "continuous integration", "continuous deployment")),
    SkillRule("C#", ("C#", "C sharp")),
    SkillRule("C++", ("C++", "CPP")),
    SkillRule("CSS", ("CSS", "CSS3")),
    SkillRule("Cypress", ("Cypress",)),
    SkillRule("Databricks", ("Databricks",)),
    SkillRule("dbt", ("dbt",)),
    SkillRule("Django", ("Django",)),
    SkillRule("Docker", ("Docker", "containerization")),
    SkillRule("Elasticsearch", ("Elasticsearch", "ElasticSearch", "Elastic Search")),
    SkillRule("ETL", ("ETL", "ELT", "data pipelines", "data pipeline")),
    SkillRule("Excel", ("Excel", "Microsoft Excel")),
    SkillRule("FastAPI", ("FastAPI",)),
    SkillRule("Feature Engineering", ("feature engineering",)),
    SkillRule("Figma", ("Figma",)),
    SkillRule("Flask", ("Flask",)),
    SkillRule("GCP", ("GCP", "Google Cloud Platform", "Google Cloud")),
    SkillRule("Git", ("Git", "GitHub", "GitLab", "Bitbucket")),
    SkillRule("Go", ("Golang",)),
    SkillRule("GraphQL", ("GraphQL",)),
    SkillRule("HTML", ("HTML", "HTML5")),
    SkillRule("Java", ("Java",)),
    SkillRule("JavaScript", ("JavaScript", "JS", "ECMAScript")),
    SkillRule("Jenkins", ("Jenkins",)),
    SkillRule("Jest", ("Jest",)),
    SkillRule("Jira", ("Jira",)),
    SkillRule("Jupyter", ("Jupyter", "Jupyter Notebook", "JupyterLab")),
    SkillRule("Kubernetes", ("Kubernetes", "K8s")),
    SkillRule("LangChain", ("LangChain",)),
    SkillRule("Linux", ("Linux", "Unix")),
    SkillRule("Looker", ("Looker", "LookML")),
    SkillRule("Machine Learning", ("machine learning", "ML models", "ML model")),
    SkillRule("Microservices", ("microservices", "microservice architecture")),
    SkillRule("MLflow", ("MLflow",)),
    SkillRule("MongoDB", ("MongoDB",)),
    SkillRule("Next.js", ("Next.js", "NextJS", "Next JS")),
    SkillRule("Node.js", ("Node.js", "NodeJS", "Node JS")),
    SkillRule("NoSQL", ("NoSQL",)),
    SkillRule("NumPy", ("NumPy",)),
    SkillRule("Pandas", ("Pandas",)),
    SkillRule("Playwright", ("Playwright",)),
    SkillRule("PostgreSQL", ("PostgreSQL", "Postgres")),
    SkillRule("Power BI", ("Power BI", "PowerBI")),
    SkillRule("Product Analytics", ("product analytics",)),
    SkillRule("PyTorch", ("PyTorch",)),
    SkillRule("Python", ("Python",)),
    SkillRule("RAG", ("RAG", "retrieval augmented generation", "retrieval-augmented generation")),
    SkillRule("React", ("React", "React.js", "ReactJS")),
    SkillRule("React Native", ("React Native",)),
    SkillRule("Redis", ("Redis",)),
    SkillRule("Regression Testing", ("regression testing",)),
    SkillRule("Ruby", ("Ruby", "Ruby on Rails", "Rails")),
    SkillRule("Rust", ("Rust",)),
    SkillRule("Salesforce", ("Salesforce",)),
    SkillRule("Selenium", ("Selenium",)),
    SkillRule("Snowflake", ("Snowflake",)),
    SkillRule("SQL", ("SQL", "T-SQL", "PL/SQL")),
    SkillRule("Tableau", ("Tableau",)),
    SkillRule("Terraform", ("Terraform",)),
    SkillRule("TensorFlow", ("TensorFlow",)),
    SkillRule("TypeScript", ("TypeScript", "TS")),
    SkillRule("Unit Testing", ("unit testing", "unit tests", "JUnit", "pytest")),
    SkillRule("UX Research", ("UX research", "user research", "usability testing")),
    SkillRule("Vue.js", ("Vue.js", "VueJS", "Vue JS")),
)


_BOUNDARY_CHARS = r"A-Za-z0-9+#/"


def _alias_pattern(alias: str) -> str:
    """Build a strict phrase pattern while allowing flexible whitespace."""
    escaped = re.escape(alias.strip())
    return escaped.replace(r"\ ", r"\s+")


@lru_cache(maxsize=1)
def _compiled_rules() -> tuple[tuple[str, re.Pattern[str]], ...]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for rule in SKILL_RULES:
        aliases = sorted(rule.aliases, key=len, reverse=True)
        pattern = "|".join(_alias_pattern(alias) for alias in aliases)
        compiled.append(
            (
                rule.canonical,
                re.compile(
                    rf"(?<![{_BOUNDARY_CHARS}])(?:{pattern})(?![{_BOUNDARY_CHARS}])",
                    re.IGNORECASE,
                ),
            )
        )
    return tuple(compiled)


def extract_skill_tags(text: str | None) -> list[str]:
    """Extract canonical skill tags from job text in first-appearance order."""
    if not text:
        return []

    found: list[tuple[int, int, str]] = []
    for rule_index, (canonical, pattern) in enumerate(_compiled_rules()):
        match = pattern.search(text)
        if match:
            found.append((match.start(), rule_index, canonical))

    found.sort(key=lambda item: (item[0], item[1]))
    return [canonical for _, _, canonical in found]
