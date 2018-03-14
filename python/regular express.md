* test website [pythex](https://pythex.org/)

### basic cheat sheet
* Special characters
    ```
    \	escape special characters
    .	matches any character
    ^	matches beginning of string
    $	matches end of string
    [5b-d]	matches any chars '5', 'b', 'c' or 'd'
    [^a-c6]	matches any char except 'a', 'b', 'c' or '6'
    R|S	matches either regex R or regex S
    ()	creates a capture group and indicates precedence
    ```
* Special sequences
    ```
    \A	start of string
    \b	matches empty string at word boundary (between \w and \W)
    \B	matches empty string not at word boundary
    \d	digit
    \D	non-digit
    \s	whitespace: [ \t\n\r\f\v]
    \S	non-whitespace
    \w	alphanumeric: [0-9a-zA-Z_]
    \W	non-alphanumeric
    \Z	end of string
    \g<id>	matches a previously defined group
    ```
* Quantifiers
    ```
    *	0 or more (append ? for non-greedy)
    +	1 or more (append ? for non-greedy)
    ?	0 or 1 (append ? for non-greedy)
    {m}	exactly mm occurrences
    {m, n}	from m to n. m defaults to 0, n to infinity
    {m, n}?	from m to n, as few as possible
    ```
* Special sequences
    ```
    (?iLmsux)	matches empty string, sets re.X flags
    (?:...)	non-capturing version of regular parentheses
    (?P...)	matches whatever matched previously named group
    (?P=)	digit
    (?#...)	a comment; ignored
    (?=...)	lookahead assertion: matches without consuming
    (?!...)	negative lookahead assertion
    (?<=...)	lookbehind assertion: matches if preceded
    (?<!...)	negative lookbehind assertion
    (?(id)yes|no)	match 'yes' if group 'id' matched, else 'no'
    ```

### sample code
* pattarn
    * sample1

        regular express: `("chrome_id":[0-9]+,"name":"[0-9a-zA-Z ]+")`

        python code: `re.findall(r'("chrome_id":[0-9]+,"name":"[0-9a-zA-Z ]+")',the_sample_text)`

        sample text: `":[{"overrides":{"gallery_refresh_frequency":1,"gallery_mobile_interstitial_frequency":4}}]}],"id":"59fa06ce5030a27305bf2aae"}],"-416346929":[[{"chrome_id":374203,"name":"Renegade 4WD 4dr Sport","dataset":{"configuration":{"style":{"modelYear":2015}}}},{"chrome_id":374202,"name":"Renegade FWD 4dr Limited","dataset":{"configuration":{"style":{"modelYear":2015}}}},{"chrome_id":374200,"name":"Renegade FWD 4dr Sport","dataset":{"configuration":{"style":{"modelYear":2015}}}},{"chrome_id":374204,"name":"Renegade 4WD 4dr Latitude","dataset":{"configuration":{"style":{"modelYear":2015}}}},`