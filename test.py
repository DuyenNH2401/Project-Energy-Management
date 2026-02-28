import os
import sys
import traci
import random
import time

# --- CONFIGURATION ---
SUMO_CONFIG = "maps/TestMap/osm.sumocfg" 
TEST_VTYPE = "test_hev_type"
TEST_VEH_ID = "calibration_car"

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

def get_passenger_edges():
    all_edges = traci.edge.getIDList()
    valid_edges = []
    for edge_id in all_edges:
        if edge_id.startswith(":"): continue
        lane_id = f"{edge_id}_0"
        try:
            allowed = traci.lane.getAllowed(lane_id)
            if not allowed or "passenger" in allowed:
                valid_edges.append(edge_id)
        except:
            continue
    return valid_edges

def run_calibration():
    print("Starting SUMO for calibration...")
    traci.start([
        "sumo", "-c", SUMO_CONFIG,
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--device.emissions.probability", "1.0",
    ])

    print(f"Defining HEV Type: {TEST_VTYPE}")
    traci.vehicletype.copy("DEFAULT_VEHTYPE", TEST_VTYPE)
    
    # --- CRITICAL CHANGE 1: Use Energy Model ---
    # This ensures getElectricityConsumption returns Physics-based values (Wh/s)
    traci.vehicletype.setEmissionClass(TEST_VTYPE, "Energy/unknown") 

    def _get_hev_consumption():
        # 1. Get Physics Data (Energy required at wheels)
        # Because we used "Energy/unknown", this returns the actual load (Wh/s)
        physics_elec_demand = traci.vehicle.getElectricityConsumption(TEST_VEH_ID) 
        
        # 2. Derive Potential Fuel (The "Parallel" Universe)
        # If we were to use the Engine for this load, how much fuel would it take?
        # Conversion: 1 Wh at wheels ~= 300 mg Gasoline (assuming ~28% ICE efficiency)
        if physics_elec_demand > 0:
            raw_fuel_potential = physics_elec_demand * 300.0
        else:
            raw_fuel_potential = 0.0 # Engine burns 0 when coasting/braking

        # 3. Get Kinematics
        speed = traci.vehicle.getSpeed(TEST_VEH_ID)
        accel = traci.vehicle.getAcceleration(TEST_VEH_ID)

        # 4. Your EMS Logic
        ICE_THRESHOLD_SPEED = 8.3 
        ICE_THRESHOLD_ACCEL = 1.0 

        real_fuel = 0.0
        real_elec = 0.0

        if physics_elec_demand < 0:
            # CASE A: Regenerative Braking (Physics says we are generating energy)
            real_fuel = 0.0
            real_elec = physics_elec_demand # Negative value
            
        elif speed < ICE_THRESHOLD_SPEED and accel < ICE_THRESHOLD_ACCEL:
            # CASE B: EV Mode (Use Battery)
            real_fuel = 0.0
            real_elec = physics_elec_demand # Positive value
            
        else:
            # CASE C: Engine Mode (Use Gas)
            real_fuel = raw_fuel_potential # Burn the derived fuel amount
            real_elec = 0.0 

        return real_fuel, real_elec

    # --- HEV CONFIGURATION ---
    # We still keep these to ensure SUMO knows it has a battery
    params = {
        "has.emissions.device": "true",
        "has.battery.device": "true",
        "has.elecHybrid.device": "true", # We keep this for future internal logic if needed
        "device.battery.capacity": "2000.00",
        "device.battery.charge": "2000.00",  
    }
    for key, val in params.items():
        traci.vehicletype.setParameter(TEST_VTYPE, key, val)

    # --- SPAWNING LOGIC (Same as before) ---
    drivable_edges = get_passenger_edges()
    spawned = False
    attempt = 0
    while not spawned and attempt < 100:
        try:
            traci.simulationStep()
            e1 = random.choice(drivable_edges)
            e2 = random.choice(drivable_edges)
            if e1 != e2:
                route = traci.simulation.findRoute(e1, e2, TEST_VTYPE)
                if route.edges and len(route.edges) > 5:
                    traci.route.add("test_route", route.edges)
                    traci.vehicle.add(TEST_VEH_ID, "test_route", typeID=TEST_VTYPE)
                    traci.vehicle.setSpeedMode(TEST_VEH_ID, 0) 
                    spawned = True
        except:
            pass
        attempt += 1

    if not spawned:
        traci.close()
        return

    print(f"Running Logic Test...")

    max_fuel = 0.0
    max_elec = 0.0
    min_elec = 0.0

    # Run for 500 steps
    for i in range(500):
        traci.simulationStep()
        
        if TEST_VEH_ID not in traci.vehicle.getIDList():
            break

        # STRESS TEST 
        curr_speed = traci.vehicle.getSpeed(TEST_VEH_ID)
        
        if i < 200:
            target = curr_speed + 2.0
            traci.vehicle.setSpeed(TEST_VEH_ID, target)
        elif i < 300:
            target = max(0, curr_speed - 2.0)
            traci.vehicle.setSpeed(TEST_VEH_ID, target)
        else:
            traci.vehicle.setSpeed(TEST_VEH_ID, 0)

        # CAPTURE DATA
        f, e = _get_hev_consumption()
        
        if f > max_fuel: max_fuel = f
        if e > max_elec: max_elec = e
        if e < min_elec: min_elec = e 

        # Print only occasionally
        if i % 1 == 0:
            status = "ICE" if f > 0 else "EV "
            if e < 0: status = "REG"
            print(f"Step {i:03} | {status} | Spd:{curr_speed:4.1f} | Fuel:{f:6.1f} | Elec:{e:5.2f}")

    print("\n" + "="*40)
    print("CALIBRATION RESULTS (Derived)")
    print("="*40)
    print(f"Peak Fuel (Calc): {max_fuel:.2f} mg/s")
    print(f"Peak Elec (Phys): {max_elec:.2f} Wh/s")
    print(f"Max Regen:        {min_elec:.2f} Wh/s")
    print("="*40)

    traci.close()


import os
import sys
import traci
import random
import time

# --- CONFIGURATION ---
SUMO_CONFIG = "maps/TestMap/osm.sumocfg" 
TEST_VTYPE = "test_hev_type"
TEST_VEH_ID = "calibration_car"

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

def get_passenger_edges():
    """Filters map for edges that allow passenger cars."""
    all_edges = traci.edge.getIDList()
    valid_edges = []
    for edge_id in all_edges:
        if edge_id.startswith(":"): continue # Skip internal junction edges
        
        # Check permission on lane 0
        lane_id = f"{edge_id}_0"
        try:
            allowed = traci.lane.getAllowed(lane_id)
            # If allowed is empty, it allows all. If 'passenger' is in list, it's valid.
            if not allowed or "passenger" in allowed:
                valid_edges.append(edge_id)
        except:
            continue
    return valid_edges

def run_calibration():
    print("Starting SUMO for calibration...")
    # Using sumo (headless) to avoid GUI errors during stress test
    traci.start([
        "sumo", "-c", SUMO_CONFIG,
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--device.emissions.probability", "1.0",
        #"--delay", "100"
    ])

    print(f"Defining HEV Type: {TEST_VTYPE}")
    traci.vehicletype.copy("DEFAULT_VEHTYPE", TEST_VTYPE)
    traci.vehicletype.setEmissionClass(TEST_VTYPE, "MMPEVEM") #HBEFA3/PC_G_EU4 #MMPEVEM #Energy/unknown
    def _get_hev_consumption():
        # 1. Get Raw Data (SUMO calculates both as if they are independent)
        raw_fuel = traci.vehicle.getFuelConsumption(TEST_VEH_ID) # mg/s
        raw_elec = traci.vehicle.getElectricityConsumption(TEST_VEH_ID) # Wh/s
        print(raw_fuel, raw_elec)
        speed = traci.vehicle.getSpeed(TEST_VEH_ID) # m/s
        accel = traci.vehicle.getAcceleration(TEST_VEH_ID) # m/s^2

        # 2. Define HEV Thresholds (Toyota Prius style)
        # Engine turns on if speed > 30 km/h (~8.3 m/s) OR high acceleration
        ICE_THRESHOLD_SPEED = 8.3 
        ICE_THRESHOLD_ACCEL = 1.0 

        real_fuel = 0.0
        real_elec = 0.0

        # 3. Apply Logic (The EMS)
        if raw_elec < 0:
            # CASE A: Regenerative Braking
            # Engine is OFF. Battery is Charging.
            real_fuel = 0.0
            real_elec = raw_elec # Keep the negative value (charging)
            
        elif speed < ICE_THRESHOLD_SPEED and accel < ICE_THRESHOLD_ACCEL:
            # CASE B: EV Mode (Creeping / City Driving)
            # Engine is OFF. Battery is Draining.
            real_fuel = 0.0
            real_elec = raw_elec
            
        else:
            # CASE C: Hybrid/Engine Mode (Highway / Hard Accel)
            # Engine is ON. Battery is assisting (or doing nothing).
            # In a real Prius, the battery might assist, but for Simplicity:
            # We assume Engine takes the load.
            real_fuel = raw_fuel
            real_elec = 0.0 
            # Optional: If you want 'Boost', keep some percentage of real_elec

        return real_fuel, real_elec
    # --- HEV CONFIGURATION ---
    params = {
        "has.emissions.device": "true",
        "has.battery.device": "true",
        "has.elecHybrid.device": "true",
        "device.battery.capacity": "20000.00",
        "device.battery.charge": "20000.00",  
        "device.elecHybrid.minBatteryCharge": "0.10",
        "device.elecHybrid.mrecBatteryCharge": "0.90"
    }
    for key, val in params.items():
        traci.vehicletype.setParameter(TEST_VTYPE, key, val)

    # --- SCAN FOR VALID ROADS ---
    print("Scanning map for drivable roads...")
    drivable_edges = get_passenger_edges()
    print(f"Found {len(drivable_edges)} valid edges.")

    spawned = False
    attempt = 0
    while not spawned and attempt < 100:
        try:
            traci.simulationStep()
            e1 = random.choice(drivable_edges)
            e2 = random.choice(drivable_edges)
            
            if e1 != e2:
                # Ask SUMO to find a path
                route = traci.simulation.findRoute(e1, e2, TEST_VTYPE)
                if route.edges and len(route.edges) > 5:
                    traci.route.add("test_route", route.edges)
                    traci.vehicle.add(TEST_VEH_ID, "test_route", typeID=TEST_VTYPE)
                    traci.vehicle.setSpeedMode(TEST_VEH_ID, 0) # Disable safety
                    spawned = True
        except:
            pass
        attempt += 1

    if not spawned:
        print("Failed to spawn vehicle. Map might be too small or disconnected.")
        traci.close()
        return

    print(f"Vehicle spawned on {e1}. Running stress test...")

    max_fuel = 0.0
    max_elec = 0.0
    min_elec = 0.0

    # Run for 500 steps
    for i in range(500):
        traci.simulationStep()
        
        if TEST_VEH_ID not in traci.vehicle.getIDList():
            print("Vehicle arrived or disappeared.")
            break

        # STRESS TEST LOGIC
        curr_speed = traci.vehicle.getSpeed(TEST_VEH_ID)
        
        # 1. HARD ACCEL (0-200 steps)
        if i < 200:
            target = curr_speed + 2.0
            traci.vehicle.setSpeed(TEST_VEH_ID, target)
        # 2. HARD BRAKE (200-300 steps)
        elif i < 300:
            target = max(0, curr_speed - 2.0)
            traci.vehicle.setSpeed(TEST_VEH_ID, target)
        # 3. IDLE (300+ steps)
        else:
            traci.vehicle.setSpeed(TEST_VEH_ID, 0)

        # CAPTURE DATA
        f, e = _get_hev_consumption()
        
        if f > max_fuel: max_fuel = f
        if e > max_elec: max_elec = e
        if e < min_elec: min_elec = e # Capture Negative (Regen)

        if i % 1 == 0:
            print(f"Step {i} | Spd: {curr_speed:.1f} | Fuel: {f:.1f} | Elec: {e:.2f}")

    print("\n" + "="*40)
    print("VERIFICATION RESULTS")
    print("="*40)
    print(f"Peak Fuel Consumption: {max_fuel:.2f} mg/s")
    print(f"Peak Elec Consumption: {max_elec:.2f} Wh/s")
    print(f"Max Regen (Braking):   {min_elec:.2f} Wh/s")
    print("-" * 40)
    
    # Recommendation Logic
    # We add 10% buffer to the observed max
    rec_fuel = max_fuel * 1.1
    rec_elec = max_elec * 1.1 # Default to 50 if 0 observed
    
    print("RECOMMENDED CONSTANTS FOR YOUR ENV:")
    print(f"MAX_FUEL = {rec_fuel:.1f}")
    print(f"MAX_ELEC = {rec_elec:.1f}")
    print("="*40)

    traci.close()

if __name__ == "__main__":
    run_calibration()
