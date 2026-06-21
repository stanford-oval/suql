#!/usr/bin/env python3
"""
Script to check GitHub repository status for issues and PRs.
This helps identify existing issues that could be addressed in contributions.
"""

import requests
import json
from typing import List, Dict, Optional

def get_github_issues(repo: str, state: str = "open") -> List[Dict]:
    """
    Fetch issues from GitHub repository.
    
    Args:
        repo: Repository in format "owner/repo"
        state: Issue state ("open", "closed", "all")
    
    Returns:
        List of issue dictionaries
    """
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {"state": state, "per_page": 100}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        issues = response.json()
        # Filter out pull requests (they appear in issues API)
        return [issue for issue in issues if "pull_request" not in issue]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching issues: {e}")
        return []

def get_github_pull_requests(repo: str, state: str = "open") -> List[Dict]:
    """
    Fetch pull requests from GitHub repository.
    
    Args:
        repo: Repository in format "owner/repo"
        state: PR state ("open", "closed", "all")
    
    Returns:
        List of PR dictionaries
    """
    url = f"https://api.github.com/repos/{repo}/pulls"
    params = {"state": state, "per_page": 100}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching pull requests: {e}")
        return []

def print_issues_summary(issues: List[Dict], title: str):
    """Print a summary of issues."""
    print(f"\n{'='*60}")
    print(f"{title} ({len(issues)} issues)")
    print(f"{'='*60}")
    
    if not issues:
        print("No issues found.")
        return
    
    for issue in issues[:10]:  # Show first 10
        labels = [label["name"] for label in issue.get("labels", [])]
        print(f"\n#{issue['number']}: {issue['title']}")
        print(f"  State: {issue['state']}")
        print(f"  Labels: {', '.join(labels) if labels else 'None'}")
        print(f"  URL: {issue['html_url']}")
        if issue.get("body"):
            body_preview = issue["body"][:100].replace("\n", " ")
            print(f"  Preview: {body_preview}...")
    
    if len(issues) > 10:
        print(f"\n... and {len(issues) - 10} more issues")

def print_prs_summary(prs: List[Dict], title: str):
    """Print a summary of pull requests."""
    print(f"\n{'='*60}")
    print(f"{title} ({len(prs)} PRs)")
    print(f"{'='*60}")
    
    if not prs:
        print("No pull requests found.")
        return
    
    for pr in prs[:10]:  # Show first 10
        labels = [label["name"] for label in pr.get("labels", [])]
        print(f"\n#{pr['number']}: {pr['title']}")
        print(f"  State: {pr['state']}")
        print(f"  Labels: {', '.join(labels) if labels else 'None'}")
        print(f"  URL: {pr['html_url']}")
        if pr.get("body"):
            body_preview = pr["body"][:100].replace("\n", " ")
            print(f"  Preview: {body_preview}...")

def main():
    """Main function to check GitHub status."""
    # Check both original and forked repositories
    repos = [
        "stanford-oval/suql",  # Original repository
        "Rakshitha-Ireddi/suql"  # Forked repository
    ]
    
    print("SUQL GitHub Repository Status Check")
    print("=" * 60)
    
    for repo in repos:
        print(f"\n\nChecking repository: {repo}")
        print("-" * 60)
        
        # Check open issues
        open_issues = get_github_issues(repo, state="open")
        print_issues_summary(open_issues, f"Open Issues in {repo}")
        
        # Check closed issues (recent)
        closed_issues = get_github_issues(repo, state="closed")
        if closed_issues:
            print_issues_summary(closed_issues[:5], f"Recent Closed Issues in {repo} (showing 5)")
        
        # Check open PRs
        open_prs = get_github_pull_requests(repo, state="open")
        print_prs_summary(open_prs, f"Open Pull Requests in {repo}")
        
        # Check closed PRs (recent)
        closed_prs = get_github_pull_requests(repo, state="closed")
        if closed_prs:
            print_prs_summary(closed_prs[:5], f"Recent Closed PRs in {repo} (showing 5)")

if __name__ == "__main__":
    main()

