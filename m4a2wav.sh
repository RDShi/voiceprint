# dir=/datanfs/vox1/dev/aac
dir=/data/vox1/wav
outputdir=/data/vox1_voicewavdata
i=0
threadings=5

for id in `ls $dir`
do
    mkdir $outputdir/$id
    for file in `ls $dir/$id`
    do
        for m4a in `ls $dir/$id/$file`
        do
            i=$[i+1]
            echo $i
            nohup ffmpeg -v quiet -i $dir/$id/$file/$m4a -ac 1 -r 16000 $outputdir/$id/$file${m4a%.*}.wav &
            if [ $i -eq $threadings ]; then
                echo $i
                wait
                i=0
            fi
        done
    done
done



