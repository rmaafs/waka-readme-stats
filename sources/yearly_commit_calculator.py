from asyncio import Semaphore, gather
from json import dumps
from re import search
from datetime import datetime
from typing import Dict, Optional, Set, Tuple

from manager_download import DownloadManager as DM
from manager_environment import EnvironmentManager as EM
from manager_github import GitHubManager as GHM
from manager_file import FileManager as FM
from manager_debug import DebugManager as DBM


# How many repositories are processed concurrently.
# Kept small on purpose: GitHub applies secondary rate limits to aggressive GraphQL concurrency.
REPO_CONCURRENCY = 4


async def calculate_commit_data(repositories: Dict) -> Tuple[Dict, Dict]:
    """
    Calculate commit data by years.
    Commit data includes contribution additions and deletions in each quarter of each recorded year.

    :param repositories: user repositories info dictionary.
    :returns: Commit quarter yearly data dictionary and commit date dictionary (`{"owner/name": {oid: committedDate}}`).
    """
    DBM.i("Calculating commit data...")
    if EM.DEBUG_RUN:
        content = FM.cache_binary("commits_data.pick", assets=True)
        if content is not None:
            DBM.g("Commit data restored from cache!")
            return tuple(content)
        else:
            DBM.w("No cached commit data found, recalculating...")

    yearly_data = dict()
    date_data = dict()
    semaphore = Semaphore(REPO_CONCURRENCY)

    async def process(ind: int, repo: Dict):
        async with semaphore:
            repo_name = "[private]" if repo["isPrivate"] else f"{repo['owner']['login']}/{repo['name']}"
            DBM.i(f"\t{ind + 1}/{len(repositories)} Retrieving repo: {repo_name}")
            await update_data_with_commit_stats(repo, yearly_data, date_data)

    await gather(*[process(ind, repo) for ind, repo in enumerate(repositories) if repo["name"] not in EM.IGNORED_REPOS])
    DBM.g("Commit data calculated!")

    if EM.DEBUG_RUN:
        FM.cache_binary("commits_data.pick", [yearly_data, date_data], assets=True)
        FM.write_file("commits_data.json", dumps([yearly_data, date_data]), assets=True)
        DBM.g("Commit data saved to cache!")
    return yearly_data, date_data


async def _get_default_branch(owner: str, name: str) -> Optional[str]:
    try:
        res = await DM.get_remote_graphql("repo_default_branch", owner=owner, name=name)
        return res["data"]["repository"]["defaultBranchRef"]["name"]
    except Exception as e:
        DBM.w(f"\t\tDefault branch unknown ({owner}/{name}): {e}")
        return None


async def _commits_ahead_of(owner: str, name: str, base: str, head: str) -> Optional[int]:
    """
    :returns: number of commits on `head` that are not reachable from `base`, or None if unknown
    (e.g. branches without a common ancestor).
    """
    try:
        res = await DM.get_remote_graphql("repo_branch_compare", owner=owner, name=name, base=base, head=head)
        return res["data"]["repository"]["ref"]["compare"]["aheadBy"]
    except Exception:
        return None


async def _collect_branch_commits(owner: str, name: str, branch: str, seen: Set[str], ahead_by: Optional[int], early_stop: bool) -> list:
    """
    Downloads the commits authored by the user on the given branch, page by page,
    skipping the commits already collected from other branches of the same repository.

    Pagination stops early (only if `early_stop` is set) when:
    - all commits of `head` that are not on the default branch were already found (`ahead_by` reached), or
    - a whole page contains no new commit (the rest of the history is shared with an already processed branch).

    :returns: list of new commit dictionaries.
    """
    new_commits = list()
    query_args = dict(owner=owner, name=name, branch=branch, id=GHM.USER.node_id)
    async for page, pagination in DM.iter_remote_graphql_pages("repo_commit_list", **query_args):
        if any(commit is None for commit in page):
            # GitHub nulls out commits whose additions/deletions it cannot compute: recover them without diff stats.
            light_page = await DM.fetch_graphql_page("repo_commit_list_light", pagination, **query_args)
            light_by_oid = {commit["oid"]: commit for commit in light_page if commit is not None}
            recovered = [dict(commit, additions=0, deletions=0) for commit in light_by_oid.values()]
            present = {commit["oid"] for commit in page if commit is not None}
            page = [commit for commit in page if commit is not None] + [commit for commit in recovered if commit["oid"] not in present]
        page_new = [commit for commit in page if commit["oid"] not in seen]
        for commit in page_new:
            seen.add(commit["oid"])
        new_commits += page_new
        if not early_stop:
            continue
        if len(page) > 0 and len(page_new) == 0:
            break
        if ahead_by is not None and len(new_commits) >= ahead_by:
            break
    return new_commits


async def update_data_with_commit_stats(repo_details: Dict, yearly_data: Dict, date_data: Dict):
    """
    Updates yearly commit data with commits from given repository.
    Every commit is counted once per repository, no matter how many branches contain it.
    Skips update if the commit isn't related to any repository.

    :param repo_details: Dictionary with information about the given repository.
    :param yearly_data: Yearly data dictionary to update.
    :param date_data: Commit date dictionary to update.
    """
    owner = repo_details["owner"]["login"]
    name = repo_details["name"]
    repo_key = f"{owner}/{name}"
    repo_name = "[private]" if repo_details.get("isPrivate") else repo_key
    try:
        branch_data = await DM.get_remote_graphql("repo_branch_list", owner=owner, name=name)
    except Exception as e:
        DBM.w(f"\t\tSkipping repo due to branch query error ({repo_name}): {e}")
        return
    if len(branch_data) == 0:
        DBM.w("\t\tSkipping repo.")
        return

    branches = [branch["name"] for branch in branch_data]
    default_branch = await _get_default_branch(owner, name)
    if default_branch in branches:
        branches.remove(default_branch)
        branches.insert(0, default_branch)

    seen: Set[str] = set()
    for branch in branches:
        ahead_by = None
        if default_branch is not None and branch != default_branch:
            ahead_by = await _commits_ahead_of(owner, name, default_branch, branch)
            if ahead_by == 0:
                continue  # fully merged branch: every commit is already reachable from the default branch
        try:
            # Early stopping is only safe once the default branch has been fully collected first.
            early_stop = default_branch is not None and branch != default_branch
            commit_data = await _collect_branch_commits(owner, name, branch, seen, ahead_by, early_stop)
        except Exception as e:
            DBM.w(f"\t\tSkipping branch due to commit query error ({repo_name}@{branch}): {e}")
            continue

        for commit in commit_data:
            date = search(r"\d+-\d+-\d+", commit["committedDate"]).group()
            curr_year = datetime.fromisoformat(date).year
            quarter = (datetime.fromisoformat(date).month - 1) // 3 + 1

            date_data.setdefault(repo_key, dict())[commit["oid"]] = commit["committedDate"]

            if repo_details["primaryLanguage"] is not None:
                language = repo_details["primaryLanguage"]["name"]
                bucket = yearly_data.setdefault(curr_year, dict()).setdefault(quarter, dict()).setdefault(language, {"add": 0, "del": 0})
                # GitHub reports null additions/deletions for commits it cannot compute the diff for.
                bucket["add"] += commit["additions"] or 0
                bucket["del"] += commit["deletions"] or 0
