cat $1 \
| grep -e "N = " -e "Total time" -e speedup \
| sed -E 's/[^0-9. ]//g' \
| sed -E 's/^[ \t]*//; s/[ \t]*$//; s/[ \t]+/ /g' > cpp-nums.txt

cat cpp-nums.txt \
| tr '\n' ' ' \
| sed 's/[ ]*$//' \
| tr ' ' ',' \
| awk '{
    count = 0;
    for (i = 1; i <= length($0); i++) {
        char = substr($0, i, 1);
        if (char == ",") {
            count++;
            if (count % 7 == 0) {
                printf "\n";
                continue;
            }
        }
        printf "%s", char;
    }
    print "";
}' \
> cpp-nums.csv