cat > /testbed/solution_patch.diff << '__SOLUTION__'
{patch}
__SOLUTION__

REVERSE=$1

if [ "$REVERSE" = "true" ] || [ "$REVERSE" = "1" ]; then
    git apply --verbose --reject --reverse /testbed/solution_patch.diff
else
    git apply --verbose --reject /testbed/solution_patch.diff
fi