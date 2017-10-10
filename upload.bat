set /p msg=Input commit message : 
git add -A
git commit -m "%msg%"
git push origin master

