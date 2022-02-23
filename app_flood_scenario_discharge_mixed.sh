#!/bin/bash -e

#-----------------------------------------------------------------------------------------
# Script information
script_name='APP - RUNNER FLOOD SCENARIO - MIXED DISCHARGE - REALTIME'
script_version="1.0.0"
script_date='2021/11/18'

virtualenv_folder='/home/cfmi.arpal.org/idro/FloodScenario/library/fp_virtualenv_python3/'
virtualenv_name='fp_virtualenv_python3_flood_libraries'
script_folder='/home/cfmi.arpal.org/idro/FloodScenario/library/fp_package_floods_application/'

# Execution example:
# python3 app_flood_scenario_main.py -settings_algorithm configuration.json -time "2020-11-02 12:00"
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Get file information
script_file='/home/cfmi.arpal.org/idro/FloodScenario/library/fp_package_floods_application/app_flood_scenario_main.py'
settings_file='/home/cfmi.arpal.org/idro/FloodScenario/flood_app_exec/app_runner_scenario_mixed/app_flood_runner_scenario_discharge_mixed_realtime.json'

# Time period execution
time_period_hour=0 # hour(s)

# Get information (-u to get gmt time)
time_now=$(date -u +"%Y-%m-%d %H:00")
#time_now="2020-10-04 00:45" # DEBUG
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# Activate virtualenv
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

# Iterate over hours
time_run=$(date -d "$time_now" +'%Y-%m-%d %H:00')
for time_period_step in $(seq 0 $time_period_hour); do
    
    # Parse time information
    time_step=$(date -d "$time_run ${time_period_step} hour ago" +'%Y-%m-%d %H:00')

	# Run python script (using setting and time)
	echo " ===> COMPUTE FLOOD SCENARIOS [${time_step}] ... "
	echo " ===> COMMAND LINE: " python $script_file -settings_file $settings_file -time $time_step

	python $script_file -settings_file $settings_file -time "$time_step"
	
	echo " ===> COMPUTE FLOOD SCENARIOS [${time_step}] ... DONE"

done

# Info script end
echo " ==> "$script_name" (Version: "$script_version" Release_Date: "$script_date")"
echo " ==> ... END"
echo " ==> Bye, Bye"
echo " ==================================================================================="
# ----------------------------------------------------------------------------------------

