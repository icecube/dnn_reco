#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import git
import pip


def get_git_infos():
    try:
        repo = git.Repo(path=os.getcwd(), search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return None, None, None, False
    sha = repo.head.object.hexsha
    short_sha = repo.git.rev_parse(sha, short=7)
    try:
        origin = repo.git.execute(["git", "remote", "get-url", "origin"])
    except git.exc.GitCommandError:
        origin = None
    uncommitted_changes = repo.is_dirty()
    return short_sha, sha, origin, uncommitted_changes


short_sha, sha, origin, uncommitted_changes = get_git_infos()

installed_packages = pip.get_installed_distributions()
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
                                  for i in installed_packages])