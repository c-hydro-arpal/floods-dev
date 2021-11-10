#!/bin/bash -e

#-----------------------------------------------------------------------------------------
# Script information
script_name='ARPAL - FLOOD SCENARIOS - RUNNER - REALTIME'
script_version="1.0.0"
script_date='2020/12/14'

# Script flag(s)
flag_download=true
flag_execution=true

# Script Info
script_folder="/home/cfmi.arpal.org/fabio.delogu/library/arpal-dev/flood/"
script_file_main="/home/cfmi.arpal.org/fabio.delogu/library/arpal-dev/flood/arpal_flood_main.py"
script_file_settings="/home/cfmi.arpal.org/fabio.delogu/flood-exec/arpal_flood_configuration.json"
script_period_execution=1 # hour(s)

# VirtualEnv Info
virtualenv_folder="/home/cfmi.arpal.org/fabio.delogu/library/virtualenv_python3/"
virtualenv_name="virtualenv_python3"

# Discharge data Info
server_remote="drift@cfmi.arpal.org"

folder_discharge_data_remote_raw="/home/drift/run_osservato_pluviometri/risultati/%YYYY%mm%dd/"
folder_discharge_data_local_raw="/home/cfmi.arpal.org/fabio.delogu/flood-ws/dynamic/source/discharge/%YYYY%mm%dd/"
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Get information (-u to get gmt time)
time_run=$(date -u +"%Y-%m-%d %H:00")
time_run="2014-11-12 10:23" # DEBUG TEST CASE
time_run="2020-12-14 10:23" # DEBUG NRT CASE
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Activate python miniconda virtualenv
export PATH=$virtualenv_folder/bin:$PATH
source activate $virtualenv_name
# Add path to pythonpath
export PYTHONPATH="${PYTHONPATH}:$script_folder"
#-----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Info script start
echo " ==================================================================================="
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> START ..."

# Iterate over days
time_run=$(date -d "$time_run" +'%Y-%m-%d %H:00')
for hour in $(seq 0 $script_period_execution); do
    
    # ----------------------------------------------------------------------------------------
    # Get time information
    time_step=$(date -d "$time_run ${hour} hour ago" +'%Y-%m-%d %H:00')
	
    year_step=$(date -u -d "$time_step" +"%Y")
    month_step=$(date -u -d "$time_step" +"%m")
    day_step=$(date -u -d "$time_step" +"%d")
    hour_step=$(date -u -d "$time_step" +"%H")
    minute_step=$(date -u -d "$time_step" +"%M")
	
    # Define remote and local folder(s)
    folder_discharge_data_remote_step=${folder_discharge_data_remote_raw/'%YYYY'/$year_step}
    folder_discharge_data_remote_step=${folder_discharge_data_remote_step/'%mm'/$month_step}
    folder_discharge_data_remote_step=${folder_discharge_data_remote_step/'%dd'/$day_step}
    folder_discharge_data_remote_step=${folder_discharge_data_remote_step/'%HH'/$hour_step}
    folder_discharge_data_remote_step=${folder_discharge_data_remote_step/'%MM'/$minute_step}
	
    folder_discharge_data_local_step=${folder_discharge_data_local_raw/'%YYYY'/$year_step}
    folder_discharge_data_local_step=${folder_discharge_data_local_step/'%mm'/$month_step}
    folder_discharge_data_local_step=${folder_discharge_data_local_step/'%dd'/$day_step}
    folder_discharge_data_local_step=${folder_discharge_data_local_step/'%HH'/$hour_step}
    folder_discharge_data_local_step=${folder_discharge_data_local_step/'%MM'/$minute_step}
    
    # Info start
    echo " ===> TIME ${time_step} ... "
	# ----------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------
    # Section to dowload datasets from remote machine to local machine
    echo " ====> DOWNLOAD DATA ... "
    if $flag_download; then

        # Create local folder
        if [ ! -d "$folder_discharge_data_local_step" ]; then
            mkdir -p $folder_discharge_data_local_step
        fi
        echo " =====> EXECUTE: scp -r drift@10.24.200.114:$folder_discharge_data_remote_step/idro*.txt ${folder_discharge_data_local_step}"
        if scp -r drift@10.24.200.114:$folder_discharge_data_remote_step/idro*.txt ${folder_discharge_data_local_step}; then
            echo " ====> DOWNLOAD DATA ... DONE"
        else
            echo " ====> DOWNLOAD DATA ... FAILED"
        fi
    else
        echo " ====> DOWNLOAD DATA ... SKIPPED. FLAG NOT ACTIVATED"
    fi
    # ----------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------
    # Section to execute the flood procedure
    echo " ====> COMPUTE FLOOD SCENARIOS ... "
    if $flag_execution; then

        # Run python command line
        echo " =====> EXECUTE: " python $script_file_main -settings_file $script_file_settings -time $time_run
        python $script_file_main -settings_file $script_file_settings -time "$time_run"

        echo " ====> COMPUTE FLOOD SCENARIOS ... DONE"

    else
        echo " ====> COMPUTE FLOOD SCENARIOS ... SKIPPED. FLAG NOT ACTIVATED"
    fi
    
    # Info end
    echo " ===> TIME ${time_step} ... DONE"
    # ----------------------------------------------------------------------------------------
    
done

# Info script end
echo " ==> ... END"
echo " ==> Bye, Bye"
echo " ==================================================================================="
# ----------------------------------------------------------------------------------------



















