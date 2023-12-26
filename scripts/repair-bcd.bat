@REM Script to repair the BCD store. This script will attempt to repair the BCD store if it is corrupt or missing. 
@REM It will also attempt to repair the boot sector if it is corrupt or missing.

bcdedit /export C:\BCD_Backup

bootrec /fixmbr

bootrec /fixboot

bootrec /scanos

bootrec rebuildbcd
