# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import os
import random
import tempfile


def shuffle_in_memory(filename_in, filename_out):
    """
    Shuffle a file, line-by-line
    """
    with open(filename_in) as fp:
        lines = fp.readlines()
    # Randomize them in place:
    random.shuffle(lines)
    # Write the new order out:
    with open(filename_out, "w") as fp:
        fp.writelines(lines)


def shuffle(filename_in, filename_out, memory_limit, file_split_count, depth=0):
    """
    Shuffle a file, recursively if needed.
    """
    if os.path.getsize(filename_in) < memory_limit:
        shuffle_in_memory(filename_in, filename_out)
    else:
        # Split the big file into smaller files
        temp_files = [
            tempfile.NamedTemporaryFile("w+", delete=False)
            for i in range(file_split_count)
        ]
        for line in open(filename_in):
            random_index = random.randint(0, len(temp_files) - 1)
            temp_files[random_index].write(line)

        # Now we shuffle each smaller file
        for temp_file in temp_files:
            temp_file.close()
            shuffle(
                temp_file.name,
                temp_file.name,
                memory_limit,
                file_split_count,
                depth + 1,
            )

        # And merge back in place of the original
        merge_files(temp_files, filename_out)


def merge_files(temp_files, filename_out):
    """
    Merge a list of file names into a single
    output filename.
    """
    with open(filename_out, "w") as fp_out:
        for temp_file in temp_files:
            with open(temp_file.name) as fp:
                line = fp.readline()
                while line:
                    fp_out.write(line)
                    line = fp.readline()
