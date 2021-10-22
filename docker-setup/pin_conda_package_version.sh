# shellcheck disable=SC2155,SC2207
​
# pins the currently installed version of a conda package and its related
# packages (e.g., numpy, numpy-base), overwriting the existing pinned versions
# if they exist
pin_package() {
    if [[ "$2" == "major" ]]; then
        local fields=1
    elif [[ "$2" == "minor" ]]; then
        local fields=1,2
    elif [[ "$2" == "exact" ]] || [ -z "$2" ]; then
        local fields="1-"
    else
        echo "bad sub-version \"${2}\"; may be one of [\"major\"|\"minor\"|\"exact\"]"
        return 1
    fi
​
    if [[ "$3" == "min" ]]; then
        local v_spec=">="
    elif [[ "$3" == "max" ]]; then
        local v_spec="<="
    elif [[ "$3" == "equal" ]] || [ -z "$3" ]; then
        local v_spec="="
    else
      echo "bad specification \"${3}\"; may be one of [\"min\"|\"max\"|\"equal\"] (default: \"equal\")"
      return 1
    fi
​
    local pkgs_to_pin=($(conda list "$1" \
        | grep "^$1" \
        | tr -s ' ' \
        | cut -d ' ' -f 1,2 \
        | cut -d '.' -f $fields \
        | sed "s/$/.*/g; s/ /$v_spec/g"))
​
    if [ "${#pkgs_to_pin[@]}" -eq 0 ]; then
      echo "no installed packages matching \"$1\""
      return 1
    fi
​
    for pkg_spec in "${pkgs_to_pin[@]}"; do
        local search_str="$(echo "$pkg_spec" | cut -d "${v_spec:0:1}" -f 1)[<>=]"
        local curr_pinned_version=$(conda config --show pinned_packages \
            | grep "^  - $search_str" \
            | sed 's/  - //')
​
        if [ -n "$curr_pinned_version" ]; then
            if [[ "$curr_pinned_version" == "$pkg_spec" ]]; then
                echo "$curr_pinned_version already pinned"
                continue
            fi
​
            conda config --remove pinned_packages "$curr_pinned_version"
            echo "unpinned $curr_pinned_version"
        fi
        conda config --add pinned_packages "$pkg_spec"
        echo "pinned $pkg_spec"
    done
}
​
​
# removes the pinned version of a package from the conda config, if one exists
unpin_package() {
    local curr_pinned=($(conda config --show pinned_packages \
        | grep "^  - $1" \
        | sed 's/  -//'))
​
    if [ "${#curr_pinned[@]}" -eq 0 ]; then
      echo "no currently pinned packages matching \"$1\""
      return 1
    fi
​
    for pkg_spec in "${curr_pinned[@]}"; do
        conda config --remove pinned_packages "$pkg_spec"
        echo "unpinned $pkg_spec"
    done
}