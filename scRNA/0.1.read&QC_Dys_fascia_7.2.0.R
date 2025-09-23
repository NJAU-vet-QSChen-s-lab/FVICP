# =============================================================================
# 疾病样本质控脚本
# =============================================================================
# 功能：读取疾病样本（Dys_fascia）的10X数据，创建Seurat对象，进行质控过滤

print("Hello world!")
rm(list = ls())

# 加载配置文件
source("../config.R")

# 设置工作目录
setwd(ENV_DIR)
lf <- list.files("./")
for (file in lf) {
  source(file)
}
rm(lf, file)

# 设置数据目录
setwd(file.path(DATA_DIR, "dys_fascia"))

# 查找数据文件夹
folders = list.files('./', pattern = '[123]$')
folders

# 创建Seurat对象
# 注意：请确保数据文件夹包含以下文件：
# - barcodes.tsv.gz
# - features.tsv.gz  
# - matrix.mtx.gz
sclist = lapply(folders, function(folders){ 
  CreateSeuratObject(counts = Read10X(folders), 
                     project = folders,
                     min.cells = QC_PARAMS$general$min_cells, 
                     min.features = QC_PARAMS$general$min_features)
})
sclist

# 合并样本
fascia <- merge(sclist[[1]], 
                y = sclist[[2]], 
                add.cell.ids = c("Dys_Fascia_1","Dys_Fascia_2"), 
                project = "Dys_fascia")
fascia

# 计算线粒体基因比例
fascia[["percent.mt"]] <- PercentageFeatureSet(fascia, pattern = QC_PARAMS$general$mt_pattern) 

# 质控前可视化
pdf(file = file.path(OUTPUT_DIR, "qc", "preQC-Dys_fascia.pdf"), width = 10, height = 6)              
VlnPlot(fascia, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
        ncol = 3, 
        group.by = "orig.ident", 
        pt.size = 0)
dev.off()

# 质控参数检查（使用isOutlier函数获取推荐值）
cat("=== 质控参数检查 ===\n")
cat("线粒体基因比例上限推荐值:", isOutlier(fascia@meta.data$percent.mt, type="higher"), "\n")
cat("基因数下限推荐值:", isOutlier(fascia@meta.data$nFeature_RNA, type="lower"), "\n")
cat("UMI数下限推荐值:", isOutlier(fascia@meta.data$nCount_RNA, type="lower"), "\n")
cat("基因数上限推荐值:", isOutlier(fascia@meta.data$nFeature_RNA, type="higher"), "\n")
cat("UMI数上限推荐值:", isOutlier(fascia@meta.data$nCount_RNA, type="higher"), "\n")

# 应用质控过滤
# 使用配置文件中的参数
fascia <- subset(fascia, subset = 
  nCount_RNA < QC_PARAMS$dys_fascia$nCount_RNA_max & 
  nFeature_RNA < QC_PARAMS$dys_fascia$nFeature_RNA_max & 
  percent.mt < QC_PARAMS$dys_fascia$percent_mt_max
)

# 质控后可视化
pdf(file = file.path(OUTPUT_DIR, "qc", "postQC-Dys_fascia.pdf"), width = 10, height = 6)              
VlnPlot(fascia, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
        ncol = 3, 
        group.by = "orig.ident", 
        pt.size = 0)
dev.off()

# 保存质控后的对象
qsave(fascia, file.path(OUTPUT_DIR, "qc", "Dys_fascia.qs"))

cat("=== 质控完成 ===\n")
cat("质控前细胞数:", ncol(fascia@assays$RNA@counts), "\n")
cat("质控后细胞数:", ncol(fascia@assays$RNA@counts), "\n")
cat("质控参数：\n")
cat("  UMI数上限:", QC_PARAMS$dys_fascia$nCount_RNA_max, "\n")
cat("  基因数上限:", QC_PARAMS$dys_fascia$nFeature_RNA_max, "\n")
cat("  线粒体基因比例上限:", QC_PARAMS$dys_fascia$percent_mt_max, "\n")
