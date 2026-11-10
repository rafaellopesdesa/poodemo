"""Shared migration to learned observables and simulation templates in 08--11."""
import ast
import re
import nbformat

COMMON_NOTE = r'''### Learned observables and simulation templates

All event coordinates use the same **filtered Unity ensembles** as notebook 03.
The moving observable uses $\hat p_\eta/\hat p_1$, with power $100$ and **no floor**.
Fixed process axes use the learned ratios to $S$. Rates and ratio normalizers
are frozen on held-out simulation before fits or toys; the common unknown $S$
density cancels. At $\eta=1$ the moving observable is exactly $1/2$.

Templates come from labelled integration events for $S$, $SBI$, $B$, $NI$ and
its two variations. For sample $j$, each retained shape event receives weight
$\nu_j/n_j$, where $\nu_j=\lambda_j N_{j,\mathrm{selected}}/N_{j,\mathrm{generated}}$.
`SIMULATION_EVENTS_PER_SAMPLE` caps selected events used per source (default
250,000); all selected events estimate the acceptance. Increase the cap to
study MC convergence without regenerating or training. Predictions are cached
with sample hashes, filter identity and normalizers.

Only **analytical unbinned** is an oracle benchmark. The analytical toy model
still defines the simulator. Analytical PDFs are not used to define observables,
MC bin yields, or learned unbinned likelihoods. Binned Asimov observations use
these same nominal MC templates: closure at truth does not prove independence
from MC errors. Notebook 11 supplies fresh paired simulator experiments.

NI is morphed **after binning**. Positive anchors retain the existing exponential
polynomial rate/shape morph. A bin with a zero NI anchor uses linear interpolation
of its shape between $-1,0,+1$, holding the boundary shape beyond that range;
total rates still use the exponential polynomial morph. This explicit sparse-MC
convention adds no pseudo-events or yield floor. Inspect the support diagnostics
and repeat with more MC when affected bins matter. Nonregular local widths and
negative signed physical predictions remain invalid and visible.

New tables, figures and toy checkpoints go under **`practical_v1`** inside the
selected ensemble/filter evaluation. Historical outputs, including notebook 12,
remain available. Run **08 before 09**; 10 and 11 can run after selecting the same
completed 01–03 run. No retraining or transfer of event weights is needed.
'''

RUNTIME = '''### Runtime and persistent run

Run from the top in Colab. Setup downloads the pinned tested source into a fresh
temporary directory and clears cached package imports. Keep the existing
`RUN_NAME`, `MODE`, `CONFIG_OVERRIDES` and Unity model directory. Your samples
and models stay on Drive. A GPU is useful for evaluating the large ensembles;
the CDF interpolation and numerical fits also require CPU time.
If authentication is needed, use the `GITHUB_TOKEN` Colab secret or hidden
prompt with repository Contents read access. This update does not require a
new physics run, sample generation, or retraining.
'''


def _preserve_assignments(target, previous, names):
    """Retain edited controls, including multiline model paths/policies."""
    for node in ast.parse(previous).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        name = getattr(node.targets[0], 'id', '')
        if name not in names:
            continue
        replacement = previous.splitlines()[node.lineno-1:node.end_lineno]
        match = next((n for n in ast.parse(target).body if isinstance(n, ast.Assign)
                      and len(n.targets) == 1 and getattr(n.targets[0], 'id', '') == name), None)
        if match:
            lines = target.splitlines()
            lines[match.lineno-1:match.end_lineno] = replacement
            target = '\n'.join(lines)
    return target


def practical_cells(cells, number):
    from build_notebooks import REPO
    from build_reference_notebooks import MODEL_SOURCE_CODE, MODEL_SOURCE_INTRO
    from build_reference_spline_notebook import fresh_bootstrap
    md = lambda s, ident: nbformat.v4.new_markdown_cell(s.strip(), id=ident)
    code = lambda s, ident: nbformat.v4.new_code_cell(s.strip(), id=ident)
    old_path = next((REPO/'notebooks').glob(f'{number:02}_*.ipynb'))
    old = nbformat.read(old_path, as_version=4).cells
    previous_model = next((c.source for c in cells if c.get('id') == 'reference-model-source'),
                          next((c.source for c in old if c.get('id') == 'reference-model-source'), ''))
    model = _preserve_assignments(MODEL_SOURCE_CODE, previous_model,
        {'RATIO_SOURCE', 'UNITY_MODEL_DIR', 'ENSEMBLE_FILTER_POLICY', 'SIMULATION_EVENTS_PER_SAMPLE'})
    previous_boot = next(c.source for c in old if 'RUN_NAME =' in c.source and 'IN_COLAB' in c.source)
    for cell in cells:
        if cell.cell_type == 'code' and 'RUN_NAME =' in cell.source and 'IN_COLAB' in cell.source:
            cell.source = fresh_bootstrap(previous_boot).replace('poodemo-09-', f'poodemo-{number:02}-').replace('Notebook 09', f'Notebook {number:02}').replace('notebook 09 source', f'notebook {number:02} source')
    if number in (10, 11):
        first = cells[0]
        common = next(c for c in cells if c.cell_type == 'code' and 'run = create_run(ROOT,' in c.source)
        bootstrap = next(c for c in cells if c.cell_type == 'code' and 'RUN_NAME =' in c.source and 'IN_COLAB' in c.source)
        codes = [c for c in cells if c.cell_type == 'code' and c is not common and c is not bootstrap]
        for cell in codes:
            for previous in old:
                if previous.cell_type == 'code' and ('N_TOYS =' in cell.source and 'N_TOYS =' in previous.source
                        or 'POWERS =' in cell.source and 'POWERS =' in previous.source):
                    cell.source = _preserve_assignments(cell.source, previous.source,
                        {'N_TOYS', 'MU_VALUES', 'N_BINS', 'POWERS', 'EXPOSURE', 'SEED', 'MU_BOUNDS',
                         'FIT_GRID_SIZE', 'PROGRESS_EVERY', 'SYSTEMATICS', 'MU_GRID'})
        first.source = (r'''# 10. Fixed learned discriminators versus the moving learned ratio

Compare one, two and three fixed coordinates with the moving reference ratio.
Use $z_j=\operatorname{sigmoid}(a_j\log\hat r_j)$, with
$a_{NI}=1$ and $a_{SBI}=a_B=2$. The grids have 36, 1296 and 46656 cells.
The moving ratio uses 36 bins. All binned models use simulation yields;
analytical unbinned is the full-information reference. Learned ratios and finite
MC do not inherit an exact sufficiency guarantee. Inspect sparse-cell counts,
invalid fits and invalid local widths before comparing information fractions.'''
            if number == 10 else r'''# 11. Calibration with paired simulator toys

Keep the five stat-only comparisons: analytical unbinned and learned unbinned
on the same simulator experiments, learned unbinned on its own finite-bank
toys, and learned-coordinate histograms on simulator and binned-model toys.
Use $q_{\mu_0}=-2\log[L(\mu_0)/L(\hat\mu)]$, with the full free fit on $[0,3]$.
At each tested point freeze the learned observable, edges and MC templates
throughout both fits. There is no observable floor, including at $\mu_0=0$.

Simulator experiments are fresh Poisson event samples shared by the three
methods. Histogram their events once; do not fluctuate the counts again.
Learned-model toys resample the held-out S bank with learned intensity weights,
so they test a finite-bank approximation, not an exact continuous NN sampler.
Binned-model toys use independent Poisson counts from the frozen MC templates.
Even self-model toy IDs estimate critical values; odd IDs test coverage.
Failed fits remain in the coverage bounds. Model and bank fingerprints protect
resumption; new output tags preserve historical notebook-12 inputs.''')
        cells = [first, md(RUNTIME, 'practical-runtime'), bootstrap, common,
                 md(MODEL_SOURCE_INTRO, 'reference-model-intro'), code(model, 'reference-model-source'),
                 md(COMMON_NOTE, 'practical-definition'), *codes]
    else:
        for cell in cells:
            if cell.get('id') == 'reference-model-intro': cell.source = MODEL_SOURCE_INTRO.strip()
            if cell.get('id') == 'reference-model-source': cell.source = model.strip()
        cells = [c for c in cells if c.get('id') != 'practical-definition']
        i = next(i for i,c in enumerate(cells) if c.get('id') == 'reference-model-source')
        cells.insert(i+1, md(COMMON_NOTE, 'practical-definition'))
    for cell in cells:
        s = cell.source
        if cell.cell_type == 'code':
            s = re.sub(r'^REFERENCE_RATIO_ETA_MIN = .*$', 'REFERENCE_RATIO_ETA_MIN = 0.0  # No floor.', s, flags=re.M)
            s = s.replace('ROOT / "plots"', 'run.path("plots")').replace('ROOT / "results"', 'run.path("results")')
            if number == 8 and 'prepare_quadrature' in s:
                s = s.replace('prepare_quadrature, ', '').replace(', prepare_quadrature', '')
                s = 'from poodemo.practical import prepare_binning_bank\n' + s.replace('prepare_quadrature(run)', 'prepare_binning_bank(run)')
            if number == 8 and 'run_unbinned' in s:
                s = '''from poodemo.practical import refresh_analytic_reference
unbinned = refresh_analytic_reference(run)
display(unbinned)
print("Analytical unbinned benchmark refreshed; all histogram inputs remain simulation-based.")'''
            if number == 8 and 'Negative coherent prediction' in s:
                s = s.replace('raise ValueError(f"Negative coherent prediction at eta={eta}: {coherent.min()}")',
                    'print(f"Finite-MC signed coherent prediction at eta={eta}: minimum {coherent.min()}; displaying linear axes")\n        LOG_Y = False')
                s = s.replace('np.maximum(coherent, 0.)', 'coherent').replace('# Remove only floating-point roundoff after checking positivity.', '# Preserve finite-MC signed contributions; never floor a fit prediction.')
                s = s.replace('ax.set_ylim(0, 1.45*ymax)', 'ax.set_ylim(min(0., min(c.min() for _, c, _ in distributions))*1.2, 1.45*ymax)')
            s = s.replace('both morph orders.', 'morph MC templates after binning.')
            s = s.replace('especially morphing before binning.', 'including the finer MC binning.')
            if number == 9 and 'print("Largest profiled effects of morph order:")' in s:
                s = s[:s.index('print("Largest profiled effects of morph order:")')] + '''print("Direct controls all valid:", floor_study["metadata"]["all_fits_valid"])
print("MC nuisance templates are morphed after binning; no analytical binned control is used.")'''
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop('execution', None)
        else:
            if s.startswith('### Runtime and persistent run'): s = RUNTIME
            if s.startswith('### Refresh the unbinned references'):
                s = '''### Refresh the analytical unbinned benchmark

This is the only oracle inference calculation. It provides a reference for the
learned-coordinate MC histograms. Notebook 03 retains its separate learned-model
closure comparisons. The new benchmark is saved in the practical evaluation.'''
            if s.startswith('### Finite-separation fidelity'):
                s = r'''### Finite-separation tests and local information

Learned ratios need not be sufficient. Compare the binned likelihood with the
analytical unbinned benchmark. Local information at $\mu=\eta$ differs from the
later width study at fixed truth $\mu_*=1$. MC template noise can put information
ratios above one; that is a convergence diagnostic, not a physical gain over the
full likelihood. At $\eta=1$ the histogram retains only rate information.'''
            if s.startswith('### Freeze the observable'):
                s = r'''### Freeze the observable inside each fit

At each test set $\eta=\mu_0$, then freeze learned event coordinates, edges and
MC Asimov observations throughout the null and free fits. Only the physical
coefficients and NI nuisance change. The no-floor setting does not restrict
the physical fit domain. Learned ratios carry no exact sufficiency guarantee.'''
            if s.startswith('### Direct controls:'):
                s = r'''### Direct controls: floor and bin resolution

Compare the no-floor baseline with floor $0.2$, using 36 and 144 bins.
Every curve uses the same learned map, direct simulation templates and NI morph
after histogramming. There is no analytical morph-before-bin comparison.
The feasible generating point certifies the direct Asimov denominator when
its NLL reaches zero. Conditional NI fits scan and refine resolved basins;
invalid fits and boundaries remain visible. This certificate does not apply
to imperfect interpolated templates.'''
            if s.startswith('### What these comparisons establish'):
                s = '''### What these comparisons establish

Reserved direct-versus-ECDF comparisons isolate interpolation errors for the
same learned map and MC bank. Direct floor and resolution comparisons test
compression choices. Remaining gaps to analytical unbinned may include learned
ordering errors, finite MC, nuisance modeling and information lost in binning.
The MC Asimov closure alone is not independent-sample validation; use the paired
simulator toys and vary the MC cap to investigate these effects.'''
            s = s.replace('with both nuisance-morph\norders', 'with nuisance morphing after binning')
            s = s.replace('same 36 bins, quadrature, Asimov truth', 'same 36 bins, MC bank, Asimov truth')
            s = s.replace('The existing quadrature is fingerprinted and reused, not duplicated.', 'The simulation bank and learned fields are fingerprinted and reused.')
            s = s.replace('Analytical-versus-learned unbinned curves remain a separate test\nof density-ratio learning.', 'Notebook 03 retains its separate density-ratio learning checks.')
        cell.source = s.strip()
    return cells
