# =============================================================================
# 细胞注释脚本
# =============================================================================
# 功能：对细胞进行注释和差异表达分析

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

# 设置项目目录
setwd(PROJECT_ROOT)

# 读取Seurat对象
fascia <- qread(file.path(OUTPUT_DIR, "seurat", "fascia.qs"))

# 创建注释目录
dir.create(file.path(OUTPUT_DIR, "annotation"), showWarnings = FALSE, recursive = TRUE)
setwd(file.path(OUTPUT_DIR, "annotation"))

# 设置样本信息
fascia$sample <- recode(fascia$orig.ident, 
                        "fascia_1" = "health", 
                        "fascia_2" = "health", 
                        "Dys_fascia_1" = "dysentery", 
                        "Dys_fascia_2" = "dysentery")

# 差异表达分析
fascia@active.ident <- fascia$seurat_clusters
workers1 = 8
maxmem = 64
cl <- detectCores()
makecore(workers1, maxmem)
cl <- makeCluster(workers1)
dge <- FindAllMarkers(fascia)
plan("sequential")
stopCluster(cl)
rm(cl)
write.csv(dge, "dge_seurat_clusters.csv")

# 定义细胞类型标记基因
cluster16Marker=c("Cd34", "Pdgfra", "Vim", "Fgfr1", "Pdgfrb", "Pecam1", "Acta2", "Lyve1", "Ddr2", "Cxcl14", "Cxcl12", "C3", "Eln", "Col3a1", "Col1a1", "Col1a2","Myoc","Piezo2",'Pla2g2a')
CP=c("Cd34", "Pdgfra")
fibroblast=c("Eng", "Nt5e", 'Thy1','Pdgfra',"Dcn")
macrophage=c("Ptprc","Itgam","Cd68","Itgax","Cd86","Cd80","Mrc1","Cd163")
langerhans=c("Ptprc","Itgam","Cd68", "Cd207")
LIPID <- c("Daglb","Lpl","Jak1","Mrc1","Cyp26b1","Aldh3a1","Apoc2","Pparg","Pltp","Fgfr4","Gsta1")
MASTCELL=c("Ptprc","Itgam","Kit","Tpsab1","Enpp3","Fcgr3a","Fcgr2a","Cd34","Cd63","Fcer1a","Itga4","Itgb7","Vcam1")
TRADITIONALTC=c("Cd34", "Pdgfra", "Pecam1")
LYMPHOIDCELLS=c("Cd3d","Cd3e","Cd3g","Ncam1","Cd19","Sdc1")
EPITHELIALCELLS=c("Cdh1","Snai1","Cdh2","Vim","Krt1")
TRPV=c("Trpv1","Trpv4","Piezo1",'Piezo2')
NERVE=c("Nes","Sox2","Cdh1","Vim","Gabarap","Apoe")
PLASMA=c("Cd38","Sdc1","Ptprc")
BASALCELL=c("Fzd2","Fzd4")
KERATINOCYTE=c("Krt5","Krt4","Krt15","Dsc3",'Krt2',"Krt10")
Melanocytes=c("Mitf", "Dct", "Tyr", "Trp1")
SEBOCYTES=c("Myc","Krt7","Muc1",'Scd',"Krt5",'Pparg',"Mc5r")
HFSC=c("Lhx2","Krt15",'Nfatc1',"Npnt")
HAIRFOLLICLE = c("Wnt10a","Krt1")
TCgenes <- c("Cd34", "Pdgfra")
ETC <- c("Hbb", "Alas2") #Hbb+ erythroid cells

# 并行绘制特征图
library(foreach)
library(doParallel)

# 设置并行后端
num_cores <- parallel::detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# 定义特征向量列表
feature_lists <- list(cluster16Marker, CP, fibroblast, macrophage, LIPID, MASTCELL, TRADITIONALTC, LYMPHOIDCELLS, EPITHELIALCELLS, TRPV, NERVE, PLASMA, BASALCELL, KERATINOCYTE, Melanocytes, SEBOCYTES, HFSC, HAIRFOLLICLE, TCgenes)
file_prefixes <- c("cluster16Marker", "CP", "fibroblast", "macrophage", "LIPID", "MASTCELL", "TRADITIONALTC", "LYMPHOIDCELLS", "EPITHELIALCELLS", "TRPV", "NERVE", "PLASMA", "BASALCELL", "KERATINOCYTE", "Melanocytes", "SEBOCYTES", "HFSC", "HAIRFOLLICLE", "TCgenes")

# 并行绘图
foreach(i = 1:length(feature_lists), .packages = c('Seurat', 'ggplot2')) %dopar% {
  features <- feature_lists[[i]]
  prefix <- file_prefixes[i]
  
  p <- VlnPlot(object = fascia, features = features, ncol = ceiling(sqrt(length(features))))
  ggsave(filename = paste0(prefix, "_vln.tiff"), 
         plot = p, 
         width = 5 * ceiling(sqrt(length(features))), 
         height = 5 * ceiling(length(features) / ceiling(sqrt(length(features)))),
         dpi = 300)
  
  p2 <- FeaturePlot(object = fascia, label = TRUE, repel = TRUE, features = features, ncol = ceiling(sqrt(length(features))))
  ggsave(filename = paste0(prefix, "_feature.tiff"), 
         plot = p2,
         width = 5 * ceiling(sqrt(length(features))), 
         height = 5 * ceiling(length(features) / ceiling(sqrt(length(features)))),
         dpi = 300)
}

stopCluster(cl)

# 细胞类型注释
fascia$celltype <- recode(fascia$seurat_clusters, 
                          "0"="CPTC", #Telocytes
                          "1"="MC", #Myeloid cells
                          "2"="EC", #Endothelial cells
                          "3"="MuC", #Mural cells
                          "4"="MC",
                          "5"="CPTC",
                          "6"="CPTC",
                          "7"="MC",
                          "8"="MC",
                          "9"="FC", #Fibroblasts
                          "10"="MC",
                          "11"="MuC",
                          "12"="LP",
                          "13"="CPTC",
                          "14"="EC",
                          "15"="MuC",
                          "16"="FC",
                          "17"="CPTC",
                          "18"="LP", #Lymphocytes
                          "19"="MuC",
                          "20"="CPTC",
                          "21"="MC",
                          "22"="MC",
                          "23"="LP",
                          "24"="CPTC",
                          "25"="EC",
                          "26"="LP",
                          "27"="ETC", #Erythrocytes
                          "28"="MC",
                          "29"="MC",
                          "30"="Others",
                          "31"="MuSC",
                          "32"="EC",
                          "33"="MC",
                          "34"="LP",
                          "35"="FC",
                          "36"="MC",
                          "37"="MuSC", #Muscle cells
                          "38"="LP",
                          "39"="FC"
                          )

# 保存第一次注释结果
qsave(fascia, file.path(OUTPUT_DIR, "seurat", "fascia.qs"))

# 精细细胞类型注释
fascia$celltype2 <- recode(fascia$seurat_clusters, 
                           "0"="CPTC",
                           "1"="MonC",
                           "2"="EC",
                           "3"="PerC",
                           "4"="MonC",
                           "5"="CPTC",
                           "6"="CPTC",
                           "7"="MonC",
                           "8"="MP",
                           "9"="FC",
                           "10"="MP",
                           "11"="MyoF",
                           "12"="LP",
                           "13"="CPTC",
                           "14"="EC",
                           "15"="PerC",
                           "16"="FC",
                           "17"="CPTC",
                           "18"="LP",
                           "19"="MyoF",
                           "20"="CPTC",
                           "21"="MAST",
                           "22"="GNC",
                           "23"="LP",
                           "24"="CPTC",
                           "25"="EC",
                           "26"="LP",
                           "27"="ETC",
                           "28"="MP",
                           "29"="MP",
                           "30"="Others",
                           "31"="MuSC",
                           "32"="EC",
                           "33"="MP",
                           "34"="LP",
                           "35"="FC",
                           "36"="GNC",
                           "37"="MuSC",
                           "38"="LP",
                           "39"="FC"
)

# 保存第二次注释结果
qsave(fascia, file.path(OUTPUT_DIR, "seurat", "fascia.qs"))

# 创建分组信息
sclist <- SplitObject(fascia, split.by = "sample")
namelist <- names(sclist)
for (i in 1:length(sclist)) {
  lt <- length(table(sclist[[i]]$celltype))
  for (j in lt) {
    sclist[[i]]$group_celltype <- paste(namelist[[i]], "_", sclist[[i]]$celltype, sep = "")
  }
  table(sclist[[i]]$group_celltype)
}
table(sclist[[1]]$group_celltype)
table(sclist[[2]]$group_celltype)
test <- merge(sclist[[1]], y = sclist[[2]])
table(test$group_celltype)
rm(sclist)

fascia$group_celltype <- test$group_celltype
rm(test)
qsave(fascia, file.path(OUTPUT_DIR, "seurat", "fascia.qs"))

# 创建celltype2分组信息
sclist <- SplitObject(fascia, split.by = "sample")
namelist <- names(sclist)
for (i in 1:length(sclist)) {
  lt <- length(table(sclist[[i]]$celltype2))
  for (j in lt) {
    sclist[[i]]$group_celltype2 <- paste(namelist[[i]], "_", sclist[[i]]$celltype2, sep = "")
  }
  table(sclist[[i]]$group_celltype2)
}
table(sclist[[1]]$group_celltype2)
table(sclist[[2]]$group_celltype2)
test <- merge(sclist[[1]], y = sclist[[2]])
table(test$group_celltype2)
rm(sclist)

fascia$group_celltype2 <- test$group_celltype2
rm(test)
qsave(fascia, file.path(OUTPUT_DIR, "seurat", "fascia.qs"))

cat("=== 细胞注释完成 ===\n")
cat("注释后的细胞类型:", unique(fascia$celltype2), "\n")
cat("样本信息:", table(fascia$sample), "\n")
