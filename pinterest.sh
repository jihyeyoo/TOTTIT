#!/bin/bash


mkdir -p My_Dataset/tone_on_tone/{beige,grey,red,blue,green}

mkdir -p My_Dataset/tone_in_tone/{pastel,vivid,muted,dark,light}



declare -A SEARCH_KEYWORDS=(

  ["beige"]="beige interior design"

  ["grey"]="grey interior design"

  ["red"]="red interior design"

  ["blue"]="blue interior design"

  ["green"]="green interior design"



  ["pastel"]="pastel room aesthetic"

  ["vivid"]="vivid interior design"

  ["muted"]="muted color interior design"

  ["dark"]="dark moody room"

  ["light"]="light bright interior design"

)


encode_query() {

  local q="$1"

  echo "${q// /%20}"

}

download_category() {

  CATEGORY=$1

  BASE_DIR=$2



  QUERY="${SEARCH_KEYWORDS[$CATEGORY]}"

  URL_ENCODED=$(encode_query "$QUERY")

  TARGET_DIR="$BASE_DIR/$CATEGORY"



  echo "-------------------------------------"

  echo "Downloading: $CATEGORY"

  echo "Query: $QUERY"

  echo "Save to: $TARGET_DIR"

  echo "-------------------------------------"

  gallery-dl \

    --range 1-100 \

    --directory "$TARGET_DIR" \

    "https://www.pinterest.com/search/pins/?q=$URL_ENCODED"

}



# ==============================

# Tone-on-tone

# ==============================

for cat in beige grey red blue green; do

  download_category "$cat" "My_Dataset/tone_on_tone"

done



# ==============================

# Tone-in-tone

# ==============================

for cat in pastel vivid muted dark light; do

  download_category "$cat" "My_Dataset/tone_in_tone"

done



echo "All downloads complete!
