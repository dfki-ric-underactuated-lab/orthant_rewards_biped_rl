if [ -d ../docs ]; then
  cp -r index.html style.css static ../docs
else
  echo "No such folder: ../docs"
fi
