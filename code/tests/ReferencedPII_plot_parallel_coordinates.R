library(MASS)
options(stringsAsFactors = FALSE)

## Parameters for the plot
data_sets <- c("EPI", "GEQ", "SHA")
data_sets <- c("EPI", "GEQ", "SHA")
seeds <- c(0, 1, 2)
p <- 4                  # number of reference profiles, do not change
max_refs_sets <-10      # minimum 10 where collected in each file, therefore change with care
                        # larger values could fail

# make an array of the sets of reference profiles that must be plotted, do not exceed 10
RS_to_plot <- c(1, 2)   

for (data_set in data_sets){
    for (seed in seeds){
        # titles, csv opening and tralala
        title <- paste("Parallel coordinate plot of", length(RS_to_plot), 
                       "sets of dominated reference profiles")
        pdf_title <- paste0("parallel_coordinate_plot_", data_set, '_seed_', seed, '_RS_',
                            paste(RS_to_plot, collapse='_', sep='_'), '.pdf')
        res_folder <- ("../res/ReferencedPII/parallel_coordinate_plots/")
        pdf_location_title <- paste0(res_folder, pdf_title)
        pdf_location_title
        pdf(pdf_location_title)
        csvfile <- paste("../res/ReferencedPII/adaptive_questioning_procedure/", data_set, 
                         "/", seed, ".csv", sep="")
        dat = read.csv(csvfile, header=FALSE)


        crit_number <- ncol(dat)

        # Initialisation off the matrix with the info to plot
        i <- RS_to_plot[1]
        RS <- dat[seq(i:p),]                                # get good lines
        RS <- matrix(as.numeric(unlist(RS)),nrow=nrow(RS))  # transform in numericmatrix
        RS <- apply(RS, 2, sort)                            # sort columns (dominated RS) 

        # add an index to manage colors
        index <- rep(i, p)
        dim(index) <- c(p,1)
        RS <- cbind(index, RS)

        # add column names for the plot
        indices <- seq(1, crit_number)
        criteria <- rep("Criteria", crit_number)
        columns_names <- c("Ref set", paste(criteria, indices))
        colnames(RS) <- columns_names

        # add the other desired RS in the matrix
        for (i in  RS_to_plot[2:length(RS_to_plot)]) {
            offset <- (p+1)*(i-1) + 1
            RSi <- dat[seq(from=offset, to=offset+p-1),]
            RSi <- matrix(as.numeric(unlist(RSi)),nrow=nrow(RSi))
            RSi <- apply(RSi, 2, sort)                            # sort columns (dominated RS) 

            index <- rep(i, p)
            dim(index) <- c(p, 1)
            RSi <- cbind(index, RSi)

            RS <- rbind(RS, RSi)
        }

        # Vector color
        my_colors_types <- c("green", "red", "blue", "black", "yellow", "orange")
        my_colors <- my_colors_types[RS[,1] %% length(my_colors_types)]
        my_colors
        parcoord(RS[,c(1:crit_number + 1)], col=my_colors, var.label=TRUE, main=title)
        dev.off()
    }
}
