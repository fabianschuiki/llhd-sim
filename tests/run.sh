#!/bin/bash
# Copyright (c) 2017 Fabian Schuiki
set -e
TESTS_DIR="$(dirname "${BASH_SOURCE[0]}")"
LLHD_SIM="$TESTS_DIR/../target/debug/llhd-sim"

while read -d $'\0' SRCFILE; do
	echo "; simulation output of $SRCFILE"
	"$LLHD_SIM" "$SRCFILE"
	echo
done < <(find $TESTS_DIR -name "*.llhd" -print0)
