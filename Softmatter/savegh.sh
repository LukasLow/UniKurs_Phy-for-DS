#!/bin/bash

cd /root/vsc_projects/PhyforDS/UniKurs_Phy-for-DS/Softmatter
git add . || { echo "Failed to add files"; exit 1; }

git commit -m "Automated commit" || { echo "Failed to commit"; exit 1; }

git push origin master || { echo "Failed to push"; exit 1; }
