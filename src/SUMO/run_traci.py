import os
import sys

# 1) Añadir las tools de SUMO al path de Python
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
else:
    raise EnvironmentError(
        "La variable de entorno SUMO_HOME no está definida. "
        "Asegúrate de que SUMO_HOME apunta a la carpeta de instalación de SUMO."
    )

import traci


def run():
    # 2) Elegir binario de SUMO
    # Usa "sumo-gui" si quieres ver la simulación, "sumo" si quieres que vaya rápido
    sumo_binary = "sumo"

    # 3) Iniciar SUMO con tu configuración
    traci.start([sumo_binary, "-c", "algeciras.sumocfg"])
    print("Simulación iniciada con TraCI")

    # 4) Parámetros de simulación
    step = 0
    max_steps = 3600  # 1 hora de simulación si step-length = 1 s

    # IDs de tus detectores (ajusta si los cambias en detectors.add.xml)
    DETECTORS = ["d_entrada_norte", "d_salida_sur"]

    # 5) Abrir archivos de salida
    veh_file = open("traci_vehiculos.csv", "w", encoding="utf-8")
    det_file = open("traci_detectores.csv", "w", encoding="utf-8")

    # Cabeceras
    veh_file.write("step,veh_id,x,y,speed,edge,lane\n")
    det_file.write("step,detector_id,nVeh,flow,occupancy,meanSpeed\n")

    try:
        # 6) Bucle principal de simulación
        while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()  # avanza un paso de simulación

            # 6.1) Info por vehículo
            veh_ids = traci.vehicle.getIDList()
            for vid in veh_ids:
                x, y = traci.vehicle.getPosition(vid)
                speed = traci.vehicle.getSpeed(vid)  # m/s
                edge_id = traci.vehicle.getRoadID(vid)
                lane_id = traci.vehicle.getLaneID(vid)
                veh_file.write(
                    f"{step},{vid},{x},{y},{speed},{edge_id},{lane_id}\n"
                )

            # 6.2) Info por sensor (induction loops)
            for det_id in DETECTORS:
                try:
                    nVeh = traci.inductionloop.getLastStepVehicleNumber(det_id)
                    flow = traci.inductionloop.getLastStepMeanSpeed(det_id)  # m/s (no es flujo)
                    # OJO: TraCI no da flow y occupancy directamente, pero
                    # podemos aproximar con variables extra si hiciera falta.
                    # Aquí usamos sólo nVeh y velocidad media para ilustrar.
                    # occupancy no está disponible directamente para E1 en TraCI.
                    meanSpeed = traci.inductionloop.getLastStepMeanSpeed(det_id)
                    # para no liar, dejamos flow=-1.0 y occupancy=-1.0 de placeholder
                    det_file.write(
                        f"{step},{det_id},{nVeh},-1.0,-1.0,{meanSpeed}\n"
                    )
                except traci.TraCIException:
                    # por si algún detector no existe o no tiene datos aún
                    det_file.write(
                        f"{step},{det_id},0,-1.0,-1.0,-1.0\n"
                    )

            step += 1

    finally:
        # 7) Cerrar todo
        veh_file.close()
        det_file.close()
        traci.close()
        print("Simulación finalizada. Datos guardados en:")
        print("  - traci_vehiculos.csv")
        print("  - traci_detectores.csv")


if __name__ == "__main__":
    run()
