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
