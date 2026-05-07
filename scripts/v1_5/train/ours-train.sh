############################ZJJversion{##############################################
#!/bin/bash
# ===== cache to big disk =====zjjadd{

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# ===== cache to big disk =====zjjadd}

# export DEVICE=localhost:0,1,2,3   # origin
: ${DEVICE:=localhost:0,1,2,3} # ZJJ
# export DEVICE=localhost:3,4
# export DEVICE="localhost:5,6"

OUTPUT_DIR_PREFIX="/media/jqzhu/APE/CODE/OTHERs/CHM_LoRA/LoRASculpt-main/Result/v3.1/llava-lora_abl_noABmask"

# export DEVICE=localhost:4
export PER_DEVICE_TRAIN_BATCH_SIZE=2  # batchsize
export GRADIENT_ACCUMULATION_STEPS=1
export DEEPSPEED_ZEROFILE="/media/jqzhu/APE/CODE/OTHERs/CHM_LoRA/LoRASculpt-main/scripts/ds_config_zero2.json" # ds_config_zero1,2,3
export NUM_TRAIN_EPOCHS=3             # epoch
export LORA_RANK=16                   # R (原64太大，改为16)
export LORA_ALPHA=16
export AB_PRESERVE_RATIO=0.1
export OMEGA=1.0
export CMR_LAMBDA=1e-3


export HOG_LAMBDA=0.2
#lora of lora
export PATCH_RANK=4   
export PATCH_ALPHA=4      
export PATCH_TRAIN_EPOCHS=0.5


#####################################################################################
export DATASET_NAME='coco'    # 'iconqa_txt', 'coco'
# export DATASET_NAME='iconqa_txt'    # 'iconqa_txt', 'coco'
export TRAINER_NAME="LoRASculpt"

HYPERPARAMS="lora-r${LORA_RANK}-a${LORA_ALPHA}-e${NUM_TRAIN_EPOCHS}-HOG${HOG_LAMBDA}-PRank${PATCH_RANK}"
export OUTPUT_DIR="${OUTPUT_DIR_PREFIX}-${DATASET_NAME}-${TRAINER_NAME}-${HYPERPARAMS}"
bash ./scripts/v1_5/train/trainconfig_lora.sh
############################ZJJversion}##############################################





##################originversion{##################
# #!/bin/bash


# export DEVICE=localhost:0,1,2,3
# OUTPUT_DIR_PREFIX="your_path_to_checkpoints/llava-v1.5-7b"


# export PER_DEVICE_TRAIN_BATCH_SIZE=4
# export GRADIENT_ACCUMULATION_STEPS=1
# export DEEPSPEED_ZEROFILE="./scripts/zero2.json"
# export NUM_TRAIN_EPOCHS=3
# export LORA_RANK=32
# export LORA_ALPHA=64
# export AB_PRESERVE_RATIO=0.1
# export OMEGA=1.0
# export CMR_LAMBDA=1e-3



# #####################################################################################
# export DATASET_NAME='iconqa_txt'    # 'iconqa_txt', 'coco'
# export TRAINER_NAME="LoRASculpt"

# HYPERPARAMS="lora-r${LORA_RANK}-a${LORA_ALPHA}-e${NUM_TRAIN_EPOCHS}-CMRLAMBDA${CMR_LAMBDA}-OMEGA${OMEGA}-RATIO${AB_PRESERVE_RATIO}"
# export OUTPUT_DIR="${OUTPUT_DIR_PREFIX}-${DATASET_NAME}-${TRAINER_NAME}-${HYPERPARAMS}"
# bash ./scripts/v1_5/train/trainconfig_lora.sh
##################originversion}##################
