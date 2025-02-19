import os
import git
import pkg_resources


def get_git_infos():
    try:
        repo = git.Repo(path=os.getcwd(), search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return None, None, None, False
    sha = str(repo.head.object.hexsha)
    short_sha = str(repo.git.rev_parse(sha, short=7))
    try:
        origin = str(
            repo.git.execute(["git", "config", "--get", "remote.origin.url"])
        )
    except git.exc.GitCommandError:
        origin = None
    uncommitted_changes = repo.is_dirty()
    return short_sha, sha, origin, uncommitted_changes


short_sha, sha, origin, uncommitted_changes = get_git_infos()

installed_packages = [
    [d.project_name, d.version] for d in pkg_resources.working_set
]


def is_newer_version(version_base, version_test):
    """Check if version_test is newer than version_base.

    Parameters
    ----------
    version_base : str
        The base version. Expected format: "major.minor.patch".
    version_test : str
        The version to test. Expected format: "major.minor.patch".

    Returns
    -------
    bool
        True if version_test is newer than version_base, False otherwise.
    """
    major_base, minor_base, patch_base = version_base.split(".")
    major_test, minor_test, patch_test = version_test.split(".")

    is_newer = False
    if major_test > major_base:
        is_newer = True
    elif major_test == major_base:
        if minor_test > minor_base:
            is_newer = True
        elif minor_test == minor_base:
            if patch_test > patch_base:
                is_newer = True
    return is_newer
