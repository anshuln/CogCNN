# for file in ./amazon_silhouette/*/*
# do
#    # echo "$file"
#    mv "$file" $(echo "$file" | sed 's/u_frame_00//g')
#    # mv "$file" $(echo "$file" | sed 's/u_frame_0//g')

# done  

# for file in ./amazon_texture/images/*/*
# do
#    # echo "$file"
#    # mv "$file" $(echo "$file" | sed 's/tex_frame_00//g')
#    mv "$file" $(echo "$file" | sed 's/u_frame_0//g')

# done  

# for file in ./images/*/*
# do
#    # echo "$file"
#    mv "$file" $(echo "$file" | sed 's/frame_00//g')
#    # mv "$file" $(echo "$file" | sed 's/u_frame_0//g')

# done  
# for d in ./greyscale/* ; do
#   for ((i=100;i>=0;i--)); do
#       if [[ -f "$d/$i.jpg" ]]; then
#           echo "Moving $d/$i.jpg"
#           let newname=$i+1;
#           if [[ $newname < 10 ]]; then
#               newname="0$newname"
#           fi
#           mv "$d/$i.jpg" "$d/$newname.jpg"
#       fi
#   done
# done
# for d in ./edges/* ; do
#   for ((i=100;i>=0;i--)); do
#       if [[ -f "$d/$i.jpg" ]]; then
#           let newname=$i+0;
#           if (( $newname < 10 )); then
#               echo "Moving $d/$i.jpg"
#               newname="0$newname"
#           fi
#           mv "$d/$i.jpg" "$d/$newname.jpg"
#       fi
#   done
# done