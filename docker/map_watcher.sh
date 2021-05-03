inotifywait -m  -r  model_output -e create -e moved_to -e close_write |
        while read dir acton file; do
                echo " The file '$file' appeared in directory '$dir'$ via '$action'"
                gsutil cp $dir/$file gs://your-bucket-name/$dir
        done

