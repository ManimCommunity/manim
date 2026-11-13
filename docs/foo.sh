commits=(
    "fd5b29c846d0975997d1cc7455b84943118f38b3" # baseline
    "38908565f165843b9a9167936b07570583a498bb" # parallize directives
)

for commit in "${commits[@]}"; do
    echo "$commit"
    git checkout -q "$commit"
    make clean > /dev/null
    echo -n "Sequential"
    time make html > /dev/null 2> /dev/null
    make clean > /dev/null
    echo -n "Parallized"
    time make html O="-j auto" > /dev/null 2> /dev/null
done
