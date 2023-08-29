arg_inner='abc'
# arg_outer='inner is jame\'s arg'
arg_outer='inner is '${arg_inner}''
echo $arg_inner
echo $arg_outer

arg_multi_blank_s='a  b  c'
arg_multi_blank_d="a  b  c"
echo $arg_multi_blank_s
echo $arg_multi_blank_d
echo "${arg_multi_blank_s}"
echo "${arg_multi_blank_d}"

spwd=`pwd`
echo $spwd