#!/bin/bash
# =============================================================================
# PyScenic 转录调控网络分析脚本
# =============================================================================
# 功能：使用PyScenic进行转录调控网络推断和调控强度分析

# 激活conda环境
conda init
conda activate pyscenic

# 设置工作目录到项目的pyscenic文件夹
PROJECT_ROOT="$(pwd)/.."
PYSCENIC_DIR="${PROJECT_ROOT}/output/pyscenic"
mkdir -p "${PYSCENIC_DIR}"
cd "${PYSCENIC_DIR}"

echo "=== 当前工作目录: $(pwd) ==="

#----------------------------------创建转换脚本
# 创建Python转换脚本
cat > trans.py << 'EOF'
import os, sys
import loompy as lp
import numpy as np
import scanpy as sc

# 获取当前目录信息
print("当前工作目录:", os.getcwd())
print("目录内容:", os.listdir(os.getcwd()))

# 读取R导出的表达矩阵
print("正在读取表达矩阵...")
x = sc.read_csv("sce_exp.csv")

# 创建loom文件的属性
row_attrs = {"Gene": np.array(x.var_names)}
col_attrs = {"CellID": np.array(x.obs_names)}

# 创建loom文件
print("正在创建loom文件...")
lp.create("sce.loom", x.X.transpose(), row_attrs, col_attrs)
print("loom文件创建完成!")
EOF

# 运行转换脚本
echo "=== 运行Python转换脚本 ==="
python trans.py
ls -la

#----------------------------------下载参考数据集
echo "=== 下载PyScenic参考数据库 ==="

# 检查并下载必要的参考文件
REF_DIR="${PYSCENIC_DIR}/references"
mkdir -p "${REF_DIR}"

# 1. feather: 全基因组排序数据库
FEATHER_FILE="${REF_DIR}/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
if [ ! -f "${FEATHER_FILE}" ]; then
    echo "下载基因组排序数据库..."
    wget -O "${FEATHER_FILE}" https://resources.aertslab.org/cistarget/databases/mus_musculus/mm10/refseq_r80/mc_v10_clust/gene_based/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather
fi

# 2. Motif到转录因子注释数据库
MOTIF_FILE="${REF_DIR}/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl"
if [ ! -f "${MOTIF_FILE}" ]; then
    echo "下载Motif注释数据库..."
    wget -O "${MOTIF_FILE}" https://resources.aertslab.org/cistarget/motif2tf/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl
fi

# 3. 转录因子列表
TF_FILE="${REF_DIR}/allTFs_mm.txt"
if [ ! -f "${TF_FILE}" ]; then
    echo "下载转录因子列表..."
    wget -O "${TF_FILE}" https://resources.aertslab.org/cistarget/tf_lists/allTFs_mm.txt
fi

# 创建软链接到工作目录
ln -sf "${FEATHER_FILE}" ./
ln -sf "${MOTIF_FILE}" ./
ln -sf "${TF_FILE}" ./

echo "=== 参考数据库下载完成 ==="

#-------------------------------------------------------------------------------
# PyScenic 三步分析流程

echo "=== 开始PyScenic分析 ==="

# 2.1 GRN (Gene Regulatory Network) 推断
echo "步骤1: 基因调控网络推断..."
pyscenic grn \
    --num_workers 8 \
    --sparse \
    --method grnboost2 \
    --output grn.csv \
    sce.loom \
    allTFs_mm.txt

# 2.2 Cistarget (Motif enrichment)
echo "步骤2: Motif富集分析..."
pyscenic ctx \
    grn.csv \
    mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather \
    --annotations_fname motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl \
    --expression_mtx_fname sce.loom \
    --mode "dask_multiprocessing" \
    --output reg.csv \
    --num_workers 8 \
    --mask_dropouts

# 2.3 AUCell (Activity scoring)
echo "步骤3: 调控子活性计算..."
pyscenic aucell \
    sce.loom \
    reg.csv \
    --output out_SCENIC.loom \
    --num_workers 8

echo "=== PyScenic分析完成 ==="
echo "输出文件位置: ${PYSCENIC_DIR}"
echo "主要输出文件:"
echo "- grn.csv: 基因调控网络"
echo "- reg.csv: 调控子信息" 
echo "- out_SCENIC.loom: 最终结果文件"

# 返回项目根目录
cd "${PROJECT_ROOT}"
