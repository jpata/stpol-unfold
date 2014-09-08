include settings.mk

all: unfoldPro neymanPro

unfoldPro: unfoldPro.cc logging.hpp loadHistogram.hpp loadFitResult.hpp scanTau.hpp
	$(CC) $(CXXFLAGS) $(LDFLAGS) -o unfoldPro unfoldPro.cc
	
neymanPro: neymanPro.cc logging.hpp loadHistogram.hpp asymmetryCalculation.hpp
	$(CC) $(CXXFLAGS) $(LDFLAGS) -o neymanPro neymanPro.cc
    

clean:
	rm -f unfoldPro neymanPro

test:
	./unfoldPro \
		--histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=mu.txt \
		--fitCovariance=mu_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=test1.root
	
	./unfoldPro \
		--histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=ele.txt \
		--fitCovariance=ele_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=test2.root

do_unfold_mu:
	mkdir -p $(OUTPUTFOLDER)
	for SYS in $(SYSLIST); do \
		./unfoldPro \
			--histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
			--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_ele.root \
			--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
			--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_tau.root \
			--fitResultPrefix=$(FITDIR) \
			--fitResult=mu.txt \
			--fitCovariance=mu_cov.root \
			--sys=$$(echo $$SYS)__up \
			--output=$(OUTPUTFOLDER)/mu__$$(echo $$SYS)__up.root \
			$(REGMODE) \
			--no-stat \
			--no-mcstat \
			--no-fiterror; \
		./unfoldPro \
			--histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
			--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_ele.root \
			--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
			--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_tau.root \
			--fitResultPrefix=$(FITDIR) \
			--fitResult=mu.txt \
			--fitCovariance=mu_cov.root \
			--sys=$$(echo $$SYS)__down \
			--output=$(OUTPUTFOLDER)/mu__$$(echo $$SYS)__down.root \
			$(REGMODE) \
			--no-stat \
			--no-mcstat \
			--no-fiterror; \
	done

	./unfoldPro \
		--histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=mu.txt \
		--fitCovariance=mu_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/mu__nominal.root 
		

	./unfoldPro \
		--histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=mu.txt \
		--fitCovariance=mu_cov.root \
		--responseMatrixName="tm__comphep" \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/mu__generator.root 

	./unfoldPro \
		--histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=mu.txt \
		--fitCovariance=mu_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/mu__stat.root \
		--no-mcstat \
		--no-fiterror

	./unfoldPro \
		--histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=mu.txt \
		--fitCovariance=mu_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/mu__mcstat.root \
		--no-stat \
		--no-fiterror

	./unfoldPro \
		--histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=mu.txt \
		--fitCovariance=mu_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/mu__fiterror.root \
		--no-stat \
		--no-mcstat

	#julia unfold.jl mu

do_unfold_combined:
	mkdir -p $(OUTPUTFOLDER)
	for SYS in $(SYSLIST); do \
		./unfoldPro \
			--histFile=$(DATADIR)/combined/merged/cos_theta_lj.root \
			--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_ele.root \
			--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_mu.root \
			--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_tau.root \
			--fitResultPrefix=$(FITDIR) \
			--fitResult=combined.txt \
			--fitCovariance=combined_cov.root \
			--sys=$$(echo $$SYS)__up \
			$(REGMODE) \
			--output=$(OUTPUTFOLDER)/combined__$$(echo $$SYS)__up.root \
			--no-stat \
			--no-mcstat \
			--no-fiterror; \
		./unfoldPro \
			--histFile=$(DATADIR)/combined/merged/cos_theta_lj.root \
			--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_ele.root \
			--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_mu.root \
			--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_tau.root \
			--fitResultPrefix=$(FITDIR) \
			--fitResult=combined.txt \
			--fitCovariance=combined_cov.root \
			--sys=$$(echo $$SYS)__down \
			$(REGMODE) \
			--output=$(OUTPUTFOLDER)/combined__$$(echo $$SYS)__down.root \
			--no-stat \
			--no-mcstat \
			--no-fiterror; \
	done

	./unfoldPro \
		--histFile=$(DATADIR)/combined/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=combined.txt \
		--fitCovariance=combined_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/combined__nominal.root 

	./unfoldPro \
		--histFile=$(DATADIR)/combined/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=combined.txt \
		--fitCovariance=combined_cov.root \
		--responseMatrixName="tm__comphep" \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/combined__generator.root 

	./unfoldPro \
		--histFile=$(DATADIR)/combined/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=combined.txt \
		--fitCovariance=combined_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/combined__stat.root \
		--no-mcstat \
		--no-fiterro

	./unfoldPro \
		--histFile=$(DATADIR)/combined/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=combined.txt \
		--fitCovariance=combined_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/combined__mcstat.root \
		--no-stat \
		--no-fiterror

	./unfoldPro \
		--histFile=$(DATADIR)/combined/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=combined.txt \
		--fitCovariance=combined_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/combined__fiterror.root \
		--no-stat \
		--no-mcstat
	#julia unfold.jl combined

do_unfold_ele:
	mkdir -p $(OUTPUTFOLDER)
	for SYS in $(SYSLIST); do \
		./unfoldPro \
			--histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
			--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
			--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_mu.root \
			--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_tau.root \
			--fitResultPrefix=$(FITDIR) \
			--fitResult=ele.txt \
			--fitCovariance=ele_cov.root \
			--sys=$$(echo $$SYS)__up \
			$(REGMODE) \
			--output=$(OUTPUTFOLDER)/ele__$$(echo $$SYS)__up.root \
			--no-stat \
			--no-mcstat \
			--no-fiterror; \
	   ./unfoldPro \
			--histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
			--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
			--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_mu.root \
			--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_tau.root \
			--fitResultPrefix=$(FITDIR) \
			--fitResult=ele.txt \
			--fitCovariance=ele_cov.root \
			--sys=$$(echo $$SYS)__down \
			$(REGMODE) \
			--output=$(OUTPUTFOLDER)/ele__$$(echo $$SYS)__down.root \
			--no-stat \
			--no-mcstat \
			--no-fiterror; \
	done

	./unfoldPro \
		--histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=ele.txt \
		--fitCovariance=ele_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/ele__nominal.root 

	./unfoldPro \
		--histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=ele.txt \
		--fitCovariance=ele_cov.root \
		--responseMatrixName="tm__comphep" \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/ele__generator.root 

	./unfoldPro \
		--histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=ele.txt \
		--fitCovariance=ele_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/ele__stat.root \
		--no-mcstat \
		--no-fiterror

	./unfoldPro \
		--histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=ele.txt \
		--fitCovariance=ele_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/ele__mcstat.root \
		--no-stat \
		--no-fiterror

	./unfoldPro \
		--histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=ele.txt \
		--fitCovariance=ele_cov.root \
		--sys=nominal \
		$(REGMODE) \
		--output=$(OUTPUTFOLDER)/ele__fiterror.root \
		--no-stat \
		--no-mcstat
	#julia unfold.jl ele

unfold_2bin_mu:
	mkdir -p $(OUTPUTFOLDER)
	for SYS in $(SYSLIST); do \
	python auto_unfold2bins.py --histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=mu.txt \
		--fitCovariance=mu_cov.root \
		--sys=$$(echo $$SYS) \
		--output=$(OUTPUTFOLDER)/mu__$$(echo $$SYS).csv; \
	done

	python auto_unfold2bins.py --histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=mu.txt \
		--fitCovariance=mu_cov.root \
		--sys=nominal \
		--responseMatrixName="tm__comphep" \
		--output=$(OUTPUTFOLDER)/mu__generator.csv

	python auto_unfold2bins.py --histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=mu.txt \
		--fitCovariance=mu_cov.root \
		--sys=nominal \
		--no-mcstat \
		--no-fiterror \
		--output=$(OUTPUTFOLDER)/mu__stat.csv

	python auto_unfold2bins.py --histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=mu.txt \
		--fitCovariance=mu_cov.root \
		--sys=nominal \
		--no-stat \
		--no-fiterror \
		--output=$(OUTPUTFOLDER)/mu__mcstat.csv

	python auto_unfold2bins.py --histFile=$(DATADIR)/mu/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/mu/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=mu.txt \
		--fitCovariance=mu_cov.root \
		--sys=nominal \
		--no-stat \
		--no-mcstat \
		--output=$(OUTPUTFOLDER)/mu__fiterror.csv
	#sed -i .bak "s/nominal/generator/" histos/mu__generator.csv
	#julia ../analysis/merge_csv.jl ../../results/tables/asymmetries_mu_2bin.csv histos/mu_*.csv

unfold_2bin_ele:
	mkdir -p $(OUTPUTFOLDER)
	for SYS in $(SYSLIST); do \
	python auto_unfold2bins.py --histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=ele.txt \
		--fitCovariance=ele_cov.root \
		--sys=$$(echo $$SYS) \
		--output=$(OUTPUTFOLDER)/ele__$$(echo $$SYS).csv; \
	done

	python auto_unfold2bins.py --histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=ele.txt \
		--fitCovariance=ele_cov.root \
		--sys=nominal \
		--responseMatrixName="tm__comphep" \
		--output=$(OUTPUTFOLDER)/ele__generator.csv

	python auto_unfold2bins.py --histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=ele.txt \
		--fitCovariance=ele_cov.root \
		--sys=nominal \
		--no-mcstat \
		--no-fiterror \
		--output=$(OUTPUTFOLDER)/ele__stat.csv

	python auto_unfold2bins.py --histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=ele.txt \
		--fitCovariance=ele_cov.root \
		--sys=nominal \
		--no-stat \
		--no-fiterror \
		--output=$(OUTPUTFOLDER)/ele__mcstat.csv

	python auto_unfold2bins.py --histFile=$(DATADIR)/ele/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/ele/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=ele.txt \
		--fitCovariance=ele_cov.root \
		--sys=nominal \
		--no-stat \
		--no-mcstat \
		--output=$(OUTPUTFOLDER)/ele__fiterror.csv
	#sed -i .bak "s/nominal/generator/" histos/ele__generator.csv
	#julia ../analysis/merge_csv.jl ../../results/tables/asymmetries_ele_2bin.csv histos/ele_*.csv

unfold_2bin_combined:
	mkdir -p $(OUTPUTFOLDER)
	for SYS in $(SYSLIST); do \
	python auto_unfold2bins.py --histFile=$(DATADIR)/combined/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=combined.txt \
		--fitCovariance=combined_cov.root \
		--sys=$$(echo $$SYS) \
		--output=$(OUTPUTFOLDER)/combined__$$(echo $$SYS).csv; \
	done

	python auto_unfold2bins.py --histFile=$(DATADIR)/combined/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=combined.txt \
		--fitCovariance=combined_cov.root \
		--sys=nominal \
		--responseMatrixName="tm__comphep" \
		--output=$(OUTPUTFOLDER)/combined__generator.csv

	python auto_unfold2bins.py --histFile=$(DATADIR)/combined/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=combined.txt \
		--fitCovariance=combined_cov.root \
		--sys=nominal \
		--no-mcstat \
		--no-fiterror \
		--output=$(OUTPUTFOLDER)/combined__stat.csv

	python auto_unfold2bins.py --histFile=$(DATADIR)/combined/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=combined.txt \
		--fitCovariance=combined_cov.root \
		--sys=nominal \
		--no-stat \
		--no-fiterror \
		--output=$(OUTPUTFOLDER)/combined__mcstat.csv

	python auto_unfold2bins.py --histFile=$(DATADIR)/combined/merged/cos_theta_lj.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_ele.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_mu.root \
		--responseFile=$(DATADIR)/combined/tmatrix_nocharge__gen_tau.root \
		--fitResultPrefix=$(FITDIR) \
		--fitResult=combined.txt \
		--fitCovariance=combined_cov.root \
		--sys=nominal \
		--no-stat \
		--no-mcstat \
		--output=$(OUTPUTFOLDER)/combined__fiterror.csv
	#sed -i .bak "s/nominal/generator/" histos/combined__generator.csv
	#julia ../analysis/merge_csv.jl ../../results/tables/asymmetries_combined_2bin.csv histos/combined_*.csv

unfold_all: do_unfold_mu do_unfold_ele unfold_2bin_mu unfold_2bin_ele do_unfold_combined unfold_2bin_combined
