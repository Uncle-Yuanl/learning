integer=$1
python $(cd `dirname $0`; pwd)/test.py -i $integer
# echo 'python脚本结果： ' $?  # 不能写，不然下一行的$?就变成了echo的系统返回值了
if [ ! $? -eq 0 ]
then
    echo 'end'
else
    echo 'interrupted'
fi

echo 'pid: ' | head -n 1 /home/yhao/learning/shell/test.py | xargs echo
head -n 1 /home/yhao/learning/shell/test.py | 'pid: ' xargs echo