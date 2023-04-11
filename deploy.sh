#! /bin/bash


# rebulid
hugo -d docs


msg="rebuilding site `date`"

if [ $# -eq 1  ]
    then msg="$1"
fi


# 
git add docs
git commit -m "$msg"
git push origin main
