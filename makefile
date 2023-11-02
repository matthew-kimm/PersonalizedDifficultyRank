.PHONY: all
all: make-completed/algebra make-completed/course make-completed/figures make-completed/tables

.PHONY: algebra
algebra: make-completed/algebra

.PHONY: course
course: make-completed/course

.PHONY: figures
figures: make-completed/figures

.PHONY: tables
tables: make-completed/tables

.PHONY: reset
reset:
	rm -rvf make-completed
	mkdir make-completed
	touch make-completed/.keep
	rm -rvf figures
	mkdir figures
	touch figures/.keep
	rm -rvf logs
	mkdir logs
	touch logs/.keep
	rm -rvf results
	mkdir results
	touch results/.keep
	rm -rvf tables
	mkdir tables
	touch tables/.keep

make-completed/algebra: tables/algebra_table.tex figures/param.pdf
	touch make-completed/algebra

make-completed/course: tables/course_table.tex
	touch make-completed/course

make-completed/figures: figures/ap.pdf figures/ndpm.pdf figures/time.pdf figures/param.pdf
	touch make-completed/figures

make-completed/tables: tables/algebra_table.tex tables/course_table.tex
	touch make-completed/tables

# Figure Requires Algebra+Course Percent Split Easiest
figures/ap.pdf: plot_figure.py results/Experiment-Result__data-set_algebra__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__easiest.csv results/Experiment-Result__data-set_course__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__easiest.csv
	python3 plot_figure.py -r ${REPEATS} -s ${SEED} -p ${PCT_TRAIN}

figures/ndpm.pdf: plot_figure.py results/Experiment-Result__data-set_algebra__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__easiest.csv results/Experiment-Result__data-set_course__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__easiest.csv
	python3 plot_figure.py -r ${REPEATS} -s ${SEED} -p ${PCT_TRAIN}

figures/time.pdf: plot_figure.py results/Experiment-Result__data-set_algebra__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__easiest.csv results/Experiment-Result__data-set_course__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__easiest.csv
	python3 plot_figure.py -r ${REPEATS} -s ${SEED} -p ${PCT_TRAIN}

# Algebra Table
tables/algebra_table.tex: results_to_table.py results/Experiment-Result__data-set_algebra__how-split_all__easiest.csv results/Experiment-Result__data-set_algebra__how-split_all__hardest.csv results/Experiment-Result__data-set_algebra__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__easiest.csv results/Experiment-Result__data-set_algebra__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__hardest.csv
	python3 results_to_table.py algebra -r ${REPEATS} -s ${SEED} -p ${PCT_TRAIN}

# Course Table
tables/course_table.tex: results_to_table.py results/Experiment-Result__data-set_course__how-split_all__hardest.csv results/Experiment-Result__data-set_course__how-split_all__easiest.csv results/Experiment-Result__data-set_course__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__easiest.csv results/Experiment-Result__data-set_course__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__hardest.csv
	python3 results_to_table.py course -r ${REPEATS} -s ${SEED} -p ${PCT_TRAIN}

# Algebra Experiments

## All Split

### Easiest
results/Experiment-Result__data-set_algebra__how-split_all__easiest.csv: data/algebra/transformed_data/algebra_easiest.csv
	python3 experiment.py algebra all 2>&1 | tee logs/algebra_all_easiest.log

### Hardest
results/Experiment-Result__data-set_algebra__how-split_all__hardest.csv: data/algebra/transformed_data/algebra_hardest.csv
	python3 experiment.py algebra all -hardest 2>&1 | tee logs/algebra_all_hardest.log

## Percent Split

### Easiest
results/Experiment-Result__data-set_algebra__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__easiest.csv: data/algebra/transformed_data/algebra_easiest.csv
	python3 experiment.py algebra percent -r ${REPEATS} -s ${SEED} -p ${PCT_TRAIN} 2>&1 | tee logs/algebra_pct_easiest.log

#### Compute Thresholds HITS-IIT (Does not select best, only computes all, set best manually in experiment.py variable hits_iit_method)
results/Experiment-Result__data-set_algebra__how-split_percent__repeats_${REPEATS}__seed_1000__pct-train_${PCT_TRAIN}__easiest__param.csv: data/algebra/transformed_data/algebra_easiest.csv
	python3 experiment.py algebra percent -r ${REPEATS} -s 1000 -p ${PCT_TRAIN} -param 2>&1 | tee logs/algebra_param.log
#### Threshold Figure
figures/param.pdf: plot_param_figure.py results/Experiment-Result__data-set_algebra__how-split_percent__repeats_${REPEATS}__seed_1000__pct-train_${PCT_TRAIN}__easiest__param.csv
	python3 plot_param_figure.py -r ${REPEATS} -s 1000 -p ${PCT_TRAIN}

### Hardest
results/Experiment-Result__data-set_algebra__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__hardest.csv: data/algebra/transformed_data/algebra_hardest.csv
	python3 experiment.py algebra percent  -r ${REPEATS} -s ${SEED} -p ${PCT_TRAIN} -hardest 2>&1 | tee logs/algebra_pct_hardest.log

# Course Experiments

## All Split

### Easiest
results/Experiment-Result__data-set_course__how-split_all__easiest.csv: data/algebra/transformed_data/algebra_easiest.csv
	python3 experiment.py course all 2>&1 | tee logs/course_all_easiest.log

### Hardest
results/Experiment-Result__data-set_course__how-split_all__hardest.csv: data/algebra/transformed_data/algebra_hardest.csv
	python3 experiment.py course all -hardest 2>&1 | tee logs/course_all_hardest.log

## Percent Split

### Easiest
results/Experiment-Result__data-set_course__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__easiest.csv: data/course/transformed_data/course_easiest.csv
	python3 experiment.py course percent -r ${REPEATS} -s ${SEED} -p ${PCT_TRAIN} 2>&1 | tee logs/course_pct_easiest.log

### Hardest
results/Experiment-Result__data-set_course__how-split_percent__repeats_${REPEATS}__seed_${SEED}__pct-train_${PCT_TRAIN}__hardest.csv: data/course/transformed_data/course_hardest.csv
	python3 experiment.py course percent -r ${REPEATS} -s ${SEED} -p ${PCT_TRAIN} -hardest 2>&1 | tee logs/course_pct_hardest.log
