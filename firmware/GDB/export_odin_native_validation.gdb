set print elements 0
set pagination off
set logging file C:/Users/OEM/Desktop/ODIN_NATIVE_RC1_Validation.txt
set logging overwrite on
set logging enabled on
p g_val_progress
p g_val_passed
p g_val_done
p *(const NativeValidationCampaign*)&g_val_campaign
set logging enabled off
