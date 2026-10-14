
MMSEQS="$1"
QUERY="$2"
BASE="$4"
DB1="$5"
DB2="$6"
DB3="$7"
USE_ENV="$8"
USE_TEMPLATES="$9"
FILTER="${10}"
TAXONOMY="${11}"
M8OUT="${12}"
DATAPATH="${13}"
EXPAND_EVAL=inf
ALIGN_EVAL=10
DIFF=3000
QSC=-20.0
MAX_ACCEPT=1000000
if [ "${FILTER}" = "1" ]; then
# 0.1 was not used in benchmarks due to POSIX shell bug in line above
#  EXPAND_EVAL=0.1
  ALIGN_EVAL=10
  QSC=0.8
  MAX_ACCEPT=100000
fi
export MMSEQS_CALL_DEPTH=1
SEARCH_PARAM="--num-iterations 3 --db-load-mode 2 -a --k-score 'seq:96,prof:80' -e 0.1 --max-seqs 10000"
FILTER_PARAM="--filter-min-enable 1000 --diff ${DIFF} --qid 0.0,0.2,0.4,0.6,0.8,1.0 --qsc 0 --max-seq-id 0.95"
EXPAND_PARAM="--expansion-mode 0 -e ${EXPAND_EVAL} --expand-filter-clusters ${FILTER} --max-seq-id 0.95"
mkdir -p "${BASE}"
colabfold_search --db1 uniref30_2103_db  --db2 pdb70_220313_db --db3 colabfold_envdb_202108_db "${QUERY}" "${DATAPATH}" "${BASE}" --mmseqs "${MMSEQS}" --use-templates 1 --db-load-mode 2

wait
