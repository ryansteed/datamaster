#!/bin/sh
#Query info: http://www.patentsview.org/api/patent.html

PATENT_NUMBER=7159248

curl -d '{"q":{"patent_number":["7159248", "7159249"] },"f":["patent_number", "patent_title"]}' -H "Content-Type: application/json" \
     -X POST http://www.patentsview.org/api/patents/query && \
    echo -e "\n -> predict OK"