import multif

def run_nozzle(w):
    # Nominal values: 28 inputs
    V_nom = [2.124000000000000e-01,
             2.269000000000000e-01,
             2.734000000000000e-01,
             3.218000000000000e-01,
             3.230000000000000e-01,
             3.343000000000000e-01,
             3.474000000000000e-01,
             4.392000000000000e-01,
             4.828000000000000e-01,
             5.673000000000000e-01,
             6.700000000000000e-01,
             3.238000000000000e-01,
             2.981000000000000e-01,
             2.817000000000000e-01,
             2.787000000000000e-01,
             2.797000000000000e-01,
             2.804000000000000e-01,
             3.36000000000000e-01,
             2.978000000000000e-01,
             3.049000000000000e-01,
             3.048000000000000e-01,
             945,
             8.96,
             60000,
             262,
             8.2e10,
             2.0e-6,
             240000]

    # Apply input w as perturbation to V_nom
    for ind in range(len(V_nom)):
        V_nom[ind] += w[ind]

    # Write the input file
    filename = "input.cfg"          # The (fixed) .cfg file
    dvname   = "inputdv_all.in"     # Write variables here
    flevel   = 0                    # Fix low fidelity

    with open(dvname, 'w') as f:
        for val in V_nom:
            f.write("{}\n".format(val))

    # Run the nozzle
    nozzle = multif.nozzle.NozzleSetup( filename, flevel );
    multif.LOWF.Run(nozzle)

    # Return results
    return nozzle.Thrust

if __name__ == "__main__":
    w = [0] * 28
    thrust = run_nozzle(w)