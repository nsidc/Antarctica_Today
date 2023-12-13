#!/usr/bin/env python
# ----------------------------------------------------------------------------
# NSIDC Data Download Script
#
# Copyright (c) 2021 Regents of the University of Colorado
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# Tested in Python 2.7 and Python 3.4, 3.6, 3.7
#
# To run the script at a Linux, macOS, or Cygwin command-line terminal:
#   $ python nsidc-data-download.py
#
# On Windows, open Start menu -> Run and type cmd. Then type:
#     python nsidc-data-download.py
#
# The script will first search Earthdata for all matching files.
# You will then be prompted for your Earthdata username/password
# and the script will download the matching files.
#
# If you wish, you may store your Earthdata username/password in a .netrc
# file in your $HOME directory and the script will automatically attempt to
# read this file. The .netrc file should have the following format:
#    machine urs.earthdata.nasa.gov login myusername password mypassword
# where 'myusername' and 'mypassword' are your Earthdata credentials.
#
from __future__ import print_function

import datetime
import math
import os.path
import sys
import time

import earthaccess

from antarctica_today.constants.paths import DATA_TB_DIR

try:
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlparse
    from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen
except ImportError:
    from urllib2 import (
        HTTPCookieProcessor,
        HTTPError,
        Request,
        URLError,
        build_opener,
        urlopen,
    )
    from urlparse import urlparse

CMR_URL = "https://cmr.earthdata.nasa.gov"
URS_URL = "https://urs.earthdata.nasa.gov"
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = (
    "{0}/search/granules.json?provider=NSIDC_ECS"
    "&sort_key[]=start_date&sort_key[]=producer_granule_id"
    "&scroll=true&page_size={1}".format(CMR_URL, CMR_PAGE_SIZE)
)


def build_version_query_params(version):
    desired_pad_length = 3
    if len(version) > desired_pad_length:
        print('Version string too long: "{0}"'.format(version))
        quit()

    version = str(int(version))  # Strip off any leading zeros
    query_params = ""

    while len(version) <= desired_pad_length:
        padded_version = version.zfill(desired_pad_length)
        query_params += "&version={0}".format(padded_version)
        desired_pad_length -= 1
    return query_params


def filter_add_wildcards(filter):
    if not filter.startswith("*"):
        filter = "*" + filter
    if not filter.endswith("*"):
        filter = filter + "*"
    return filter


def build_filename_filter(filename_filter):
    filters = filename_filter.split(",")
    result = "&options[producer_granule_id][pattern]=true"
    for filter in filters:
        result += "&producer_granule_id[]=" + filter_add_wildcards(filter)
    return result


def build_cmr_query_url(
    short_name,
    version,
    time_start,
    time_end,
    bounding_box=None,
    polygon=None,
    filename_filter=None,
):
    params = "&short_name={0}".format(short_name)
    params += build_version_query_params(version)
    params += "&temporal[]={0},{1}".format(time_start, time_end)
    if polygon:
        params += "&polygon={0}".format(polygon)
    elif bounding_box:
        params += "&bounding_box={0}".format(bounding_box)
    if filename_filter:
        params += build_filename_filter(filename_filter)
    return CMR_FILE_URL + params


def get_speed(time_elapsed, chunk_size):
    if time_elapsed <= 0:
        return ""
    speed = chunk_size / time_elapsed
    if speed <= 0:
        speed = 1
    size_name = ("", "k", "M", "G", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(speed, 1000)))
    p = math.pow(1000, i)
    return "{0:.1f}{1}B/s".format(speed / p, size_name[i])


def output_progress(count, total, status="", bar_len=60):
    if total <= 0:
        return
    fraction = min(max(count / float(total), 0), 1)
    filled_len = int(round(bar_len * fraction))
    percents = int(round(100.0 * fraction))
    bar = "=" * filled_len + " " * (bar_len - filled_len)
    fmt = "  [{0}] {1:3d}%  {2}   ".format(bar, percents, status)
    print("\b" * (len(fmt) + 4), end="")  # clears the line
    sys.stdout.write(fmt)
    sys.stdout.flush()


def cmr_read_in_chunks(file_object, chunk_size=1024 * 1024):
    """Read a file in chunks using a generator. Default chunk size: 1Mb."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


def cmr_download(urls, force=False, quiet=False, output_directory=None):
    """Download files from list of urls."""
    if not urls:
        return

    url_count = len(urls)
    if not quiet:
        print(f"Downloading {url_count} files...")

    files_saved = []

    for index, url in enumerate(urls, start=1):
        filename = url.split("/")[-1]
        if not quiet:
            print(
                "{0}/{1}: {2}".format(
                    str(index).zfill(len(str(url_count))), url_count, filename
                )
            )

        # Put the new file into the output directory where we want it.
        if output_directory:
            filename = os.path.join(output_directory, filename)

        try:
            req = Request(url)
            if credentials:
                req.add_header(f"Authorization", "Basic {credentials}")
            opener = build_opener(HTTPCookieProcessor())
            response = opener.open(req)
            length = int(response.headers["content-length"])
            try:
                if not force and length == os.path.getsize(filename):
                    if not quiet:
                        print("  File exists, skipping")
                    continue
            except OSError:
                pass
            count = 0
            chunk_size = min(max(length, 1), 1024 * 1024)
            max_chunks = int(math.ceil(length / chunk_size))
            time_initial = time.time()
            with open(filename, "wb") as out_file:
                for data in cmr_read_in_chunks(response, chunk_size=chunk_size):
                    out_file.write(data)
                    if not quiet:
                        count = count + 1
                        time_elapsed = time.time() - time_initial
                        download_speed = get_speed(time_elapsed, count * chunk_size)
                        output_progress(count, max_chunks, status=download_speed)
            if not quiet:
                print()

            files_saved.append(filename)

        except HTTPError as e:
            print(f"HTTP error {e.code}, {e.reason} ({e.url})".format(e.code, e.reason))
            raise
        except URLError as e:
            print(f"URL error: {e.reason} ({e.url})")
        except IOError:
            raise

    return files_saved


def _results_with_links(results: list) -> list:
    """Filter results to only include those with download links.

    Some NSIDC-0080 CMR results lack links, but it's OK because they're duplicates. I'm
    not sure why it's like that.
    """
    filtered = [r for r in results if r.data_links()]
    return filtered


def download_new_files(
    *,
    time_start="2021-02-17",
    time_end=datetime.datetime.now().strftime("%Y-%m-%d"),
    argv=None,
) -> list[str]:
    """Download new NSIDC-0080 files into the directory of your choice.

    Will download 25km resolution data files from the southern hemisphere.
    """
    short_name = "NSIDC-0080"
    version = "2"
    output_directory = DATA_TB_DIR / short_name.lower()

    filename_filter = "*25km_*"
    bounding_box = (-180, -90, 180, 0)

    try:
        earthaccess.login()

        # Due to a known issue in earthdata, "Z" (Zulu-time) appended at the end of our string is breaking the search.
        # Issue is documented here: https://github.com/nsidc/earthaccess/issues/330
        # We don't need time-zone information in this search, so remove it here, which seems to fix things.
        # It's a hack but it works for now. These two lines may be removed when that earthdata issue is fixed.
        time_start = time_start.rstrip("Z")
        time_end = time_end.rstrip("Z")

        results = earthaccess.search_data(
            short_name=short_name,
            version=version,
            bounding_box=bounding_box,
            temporal=(time_start, time_end),
            granule_name=filename_filter,
            debug=True,
        )
        results = _results_with_links(results)
        print(f"Found {len(results)} downloadable granules.")

        # If there are no granules to download, return an empty list of files without bothering to call "download()."
        if len(results) == 0:
            files_saved = []
        # Otherwise download the files and return the list of files we downloaded.
        else:
            files_saved = earthaccess.download(results, str(output_directory))

    except KeyboardInterrupt:
        quit()

    return files_saved


if __name__ == "__main__":
    download_new_files(time_start="2023-10-21")
