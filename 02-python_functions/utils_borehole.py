

# Diane Doolaeghe: 03/05/2022

# Functions to analyse a borehole


import sys, os, math, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats


class BoreholeAnalysis:
    '''Class for analysing boreholes and plot data
    '''
    def __init__(self, bdataframe):
        if bdataframe.empty:
            raise ReferenceError("The Borehole dataset is empty.")
        self._bdataframe = bdataframe
        self._secup = bdataframe['ADJUSTEDSECUP(m)'].min()
        self._seclow = bdataframe['ADJUSTEDSECUP(m)'].max()
        self.b_length = self._seclow-self._secup
        self.p10 = self._bdataframe.shape[0]/self.b_length
        self.angle_correction(self._bdataframe) #computed once, for missing angle values, we use the borehole average
        self.p32 = self._bdataframe['angleCorrection'].sum()/self.b_length
        #open fractures
        self._bdataframe_open = self._bdataframe.loc[self._bdataframe['FRACT_INTERPRET'].isin(['Open','Partly open'])]
        self.p10_open = self._bdataframe_open.shape[0]/self.b_length
        self.p32_open = self._bdataframe_open['angleCorrection'].sum()/self.b_length
        self.fop = self.p32_open/self.p32

        #dataframe for P32(x) fop(x) ...
        self.analyses_dataframe_c = pd.DataFrame() #classic method
        self.analyses_dataframe_s = pd.DataFrame() #smoothed method

        #lithology limit
        self._all_litho_limits = {}
    

    def get_length(self, lithology_type = '', lithology_name = 'all'):
        '''return borehole length, or if a lithology is specified, the total length of this lithology in the borehole
        '''
        if lithology_type == '':
            return self.b_length
        else:
            self.borehole_lithology_limits(lithology_type)
            if lithology_name == 'all':
                litho_length = self._all_litho_limits[lithology_type]['sec_length'].sum()
                return litho_length
            else:
                #select lithology name
                _selection = self._all_litho_limits[lithology_type].loc[self._all_litho_limits[lithology_type]['lithologies'] == lithology_name]
                litho_length = _selection['sec_length'].sum()
                return litho_length

    def get_p32(self, lithology_type = '', lithology_name = 'all'):
        if lithology_type == '':
            return self.p32
        else:
            if lithology_name == 'all':
                data_select = self._bdataframe.loc[self._bdataframe[lithology_type].notna()]
            else:
                data_select = self._bdataframe.loc[self._bdataframe[lithology_type]==lithology_name]
            p32 = data_select['angleCorrection'].sum()/self.get_length(lithology_type = lithology_type, lithology_name = lithology_name)
            return p32

    def get_p10(self, lithology_type = '', lithology_name = 'all'):
        if lithology_type == '':
            return self.p10
        else:
            if lithology_name == 'all':
                data_select = self._bdataframe.loc[self._bdataframe[lithology_type].notna()]
            else:
                data_select = self._bdataframe.loc[self._bdataframe[lithology_type]==lithology_name]
            p10 = data_select.size()/self.get_length(lithology_type = lithology_type, lithology_name = lithology_name)
            return p10


    def get_fop(self, lithology_type = '', lithology_name = 'all'):
        if lithology_type == '':
            return self.fop
        else:
            if lithology_name == 'all':
                data_select = self._bdataframe.loc[self._bdataframe[lithology_type].notna()]
            else:
                data_select = self._bdataframe.loc[self._bdataframe[lithology_type]==lithology_name]
            open_data_select = data_select.loc[data_select['FRACT_INTERPRET'].isin(['Open','Partly open'])]
            fop = open_data_select['angleCorrection'].sum()/data_select['angleCorrection'].sum()
            return fop


    def angle_correction(self, dataframe):
        '''Terzaghi correction
        '''
        uncertainty = 5  
        max_f = 1/math.sin(uncertainty*math.pi/180)
        corr = 1/(np.sin(dataframe['ALPHA(degrees)']*math.pi/180))
        corr[corr > max_f] = max_f
        #fracture without angle info 
        n_miss_angle = dataframe.loc[dataframe['ALPHA(degrees)'].isnull()].shape[0]
        #print('Missing alpha angle proportion: '+str(n_miss_angle/dataframe.shape[0]))
        mean = corr.mean()
        corr = corr.fillna(mean) #the mean of the correction is given to these fractures
        #dataframe['angleCorrection']=corr
        dataframe.loc[:,('angleCorrection')] = corr

    def compute_density_along_borehole(self, method, sampling_size, frac_type = 'all', lithology_type = '', lithology_name = 'all', orientation_bin = {}, flag = ''):
        '''method: 'classic' or 'smoothed'
        sampling_size: size of the measuring window. Bin size for the classic method, or width of the Gaussian for the 'smoothed' method (2xstd).
        frac_type: 'all', 'open', or 'sealed'
        lithology_type: type of lithology to select, e.g. FRACTURE_DOMAIN, 
        lithology_name: name of lithology, e.g. 'FFM01'
        orientation_bin: dictionnary
        other_type: other type of fracture attribute to select, e.g. MIN
        other_name: attribute name, e.g. 'Calcite'
        '''
        #--select data: all, open, or sealed
        if frac_type == 'all':
            df_sel = self._bdataframe
        elif frac_type == 'open':
            df_sel = self._bdataframe.loc[self._bdataframe['FRACT_INTERPRET'].isin(['Open','Partly open'])].copy()
            self.angle_correction(df_sel)
        elif frac_type == 'sealed':
            df_sel = self._bdataframe.loc[self._bdataframe['FRACT_INTERPRET']=='sealed'].copy()
            self.angle_correction(df_sel)
        #--select_data: lithology
        if lithology_type != '':
            if lithology_type not in self._all_litho_limits.keys():
                self.borehole_lithology_limits(lithology_type) #prepare lithology limits
            if type(lithology_name) == str and lithology_name == 'all':
                df_sel=df_sel.loc[df_sel[lithology_type].notna()]
            elif  type(lithology_name) == str and lithology_name != 'all':
                df_sel=df_sel.loc[df_sel[lithology_type] == lithology_name]
            elif type(lithology_name) is list:
                df_sel=df_sel.loc[df_sel[lithology_type].isin(lithology_name)]
        #--select data: orientation bin
        if orientation_bin != {}:
            dip_sec = orientation_bin['dips']
            strike_sec = orientation_bin['strikes']
            if len(dip_sec) == 2:
                df_sel = df_sel.loc[(df_sel['DIP(degrees)'] < dip_sec[1]) & (df_sel['DIP(degrees)'] > dip_sec[0])]
                if strike_sec[0]>strike_sec[1]: #case where the bin is containing the 360° strike angle
                    df_sel = df_sel.loc[(df_sel['STRIKE(degrees)'] < strike_sec[1]) & (df_sel['STRIKE(degrees)'] > 0) | (df_sel['STRIKE(degrees)'] > strike_sec[0]) & (df_sel['STRIKE(degrees)'] < 360)]
                else:
                    df_sel = df_sel.loc[(df_sel['STRIKE(degrees)'] < strike_sec[1]) & (df_sel['STRIKE(degrees)'] > strike_sec[0])]
            elif len(dip_sec) == 4: #two bins for vertical fractures
                df_sel = df_sel.loc[(df_sel['DIP(degrees)'] < dip_sec[1]) & (df_sel['DIP(degrees)'] > dip_sec[0]) | (df_sel['DIP(degrees)'] < dip_sec[3]) & (df_sel['DIP(degrees)'] > dip_sec[2])]
                if strike_sec[0]>strike_sec[1]:
                    df_sel = df_sel.loc[(df_sel['STRIKE(degrees)'] < strike_sec[1]) & (df_sel['STRIKE(degrees)'] > 0) | (df_sel['STRIKE(degrees)'] > strike_sec[0]) & (df_sel['STRIKE(degrees)'] < 360) | (df_sel['STRIKE(degrees)'] < strike_sec[3]) & (df_sel['STRIKE(degrees)'] > strike_sec[2])]
                elif strike_sec[2]>strike_sec[3]:
                    df_sel = df_sel.loc[(df_sel['STRIKE(degrees)'] < strike_sec[3]) & (df_sel['STRIKE(degrees)'] > 0) | (df_sel['STRIKE(degrees)'] > strike_sec[2]) & (df_sel['STRIKE(degrees)'] < 360) | (df_sel['STRIKE(degrees)'] < strike_sec[1]) & (df_sel['STRIKE(degrees)'] > strike_sec[0])]
                else:
                    df_sel = df_sel.loc[(df_sel['STRIKE(degrees)'] < strike_sec[1]) & (df_sel['STRIKE(degrees)'] > strike_sec[0]) | (df_sel['STRIKE(degrees)'] < strike_sec[3]) & (df_sel['STRIKE(degrees)'] > strike_sec[2])]


        #--secup and seclow lists in the case of a selection by lithology
        if lithology_type != '': #select some sections of the borehole
            if lithology_name == 'all':
                secup_list = self._all_litho_limits[lithology_type]['secup'].tolist()
                seclow_list = self._all_litho_limits[lithology_type]['seclow'].tolist()
            else:
                #select litho name
                _selection = self._all_litho_limits[lithology_type].loc[self._all_litho_limits[lithology_type]['lithologies'] == lithology_name]
                secup_list = _selection['secup'].tolist()
                seclow_list = _selection['seclow'].tolist()
        else: #whole borehole computation
            secup_list = [self._secup]
            seclow_list = [self._seclow]

        #--compute_densities, two methods, classic and smoothed
        if method == 'classic':
            P10 = []
            P32 = []
            x = []#adjusted secup
            z = []#elevation
            sec_data = df_sel['ADJUSTEDSECUP(m)'].values
            corr = df_sel['angleCorrection'].values
            elevation_data = df_sel['ELEVATION_ADJUSTEDSECUP'].values
            for secup, seclow in zip(secup_list, seclow_list):
                if seclow-secup < sampling_size: #special case if the section is smaller than the sampling size
                    bins = np.array([secup, seclow])
                else:
                    bins=np.arange(secup, seclow, sampling_size) # TO DO, note: seclow is excluded, make a small bin ?
                bin_min = bins[:-1]
                bin_max = bins[1:]
                for i in range(len(bin_min)):
                    index = tuple([(sec_data>bin_min[i]) & (sec_data<bin_max[i])])
                    P10.append(sec_data[index].shape[0]/(bin_max[i]-bin_min[i]))
                    P32.append(np.sum(corr[index])/(bin_max[i]-bin_min[i]))
                    if sec_data[index].shape[0] != 0:
                        x.append(np.mean(sec_data[index])) 
                        z.append(np.mean(elevation_data[index])) 
                    else:
                        x.append((bin_min[i]+bin_max[i])/2)
                        z.append(np.nan) #TODO, how to retrieve z, when there is no fractures
            #--outputs
            if frac_type == 'all':
                self.analyses_dataframe_c['p10'+str(flag)]=np.array(P10)
                self.analyses_dataframe_c['p32'+str(flag)]=np.array(P32)
                self.analyses_dataframe_c['x'+str(flag)]=np.array(x)
                self.analyses_dataframe_c['z'+str(flag)]=np.array(z)
            else: #case with open or sealed
                self.analyses_dataframe_c['p10_'+str(frac_type)+str(flag)]=np.array(P10)
                self.analyses_dataframe_c['p32_'+str(frac_type)+str(flag)]=np.array(P32)
                self.analyses_dataframe_c['x_'+str(frac_type)+str(flag)]=np.array(x)
                self.analyses_dataframe_c['z_'+str(frac_type)+str(flag)]=np.array(z)
            return self.analyses_dataframe_c

        if method == 'smoothed':
            dx = 1
            x = np.arange(self._secup, self._seclow, dx) #borehole vector, note: calculated on complete borehole data
            #--P10 and P32 vectors, filled with nan
            P10 = np.full(x.shape[0], np.nan)
            P32 = np.full(x.shape[0], np.nan)
            z = np.full(x.shape[0], np.nan)
            for secup, seclow in zip(secup_list, seclow_list):
                indices = tuple([(x>=secup) & (x<=seclow)])
                x_select = x[indices]
                #select data in section
                df_section = df_sel.loc[(df_sel['ADJUSTEDSECUP(m)'] >= secup) & (df_sel['ADJUSTEDSECUP(m)'] <= seclow)]
                fracture_data = df_section['ADJUSTEDSECUP(m)'].values
                corr = df_section['angleCorrection'].values
                w_P10 = np.empty([len(fracture_data),len(x[indices])])
                w_P32 = np.empty([len(fracture_data),len(x[indices])])
                for i in range(0,len(fracture_data)):
                    x0 = fracture_data[i]
                    #gaussian centralized on the fracture, with std = sampling_size/2
                    w_P10[i,:] = scipy.stats.norm.pdf(x[indices], x0, sampling_size/2)
                    w_P32[i,:] = scipy.stats.norm.pdf(x[indices], x0, sampling_size/2)*corr[i]
                    #normalisation by the area of the curve, if the gaussian is cutted by the delimitations of the borehole or section
                    curve = w_P10[i,:]
                    area = np.sum(curve)*dx
                    w_P10[i,:] = w_P10[i,:]/area
                    w_P32[i,:] = w_P32[i,:]/area
                P10[indices] = np.nansum(w_P10, axis=0)
                P32[indices] = np.nansum(w_P32, axis=0)
                #compute elevation vector (must preferably be computed with a lot of fractures, when P10 and P32 are computed on all data)
                elevation_data = df_section['ELEVATION_ADJUSTEDSECUP'].values
                A = w_P10*elevation_data.reshape((len(elevation_data),1)) #checked: same result if we use w_P32
                z[indices] = np.sum(A, axis=0)/np.sum(w_P10,axis=0) 
            #--outputs
            if frac_type == 'all':
                self.analyses_dataframe_s['p10'+str(flag)]=P10
                self.analyses_dataframe_s['p32'+str(flag)]=P32
                self.analyses_dataframe_s['x'+str(flag)]=x
                self.analyses_dataframe_s['z'+str(flag)]=z
            else:
                self.analyses_dataframe_s['p10_'+str(frac_type)+str(flag)]=P10
                self.analyses_dataframe_s['p32_'+str(frac_type)+str(flag)]=P32
                self.analyses_dataframe_s['x_'+str(frac_type)+str(flag)]=x
                self.analyses_dataframe_s['z_'+str(frac_type)+str(flag)]=z
            return self.analyses_dataframe_s

    def compute_fop_along_borehole(self, method, flag = ''):
        '''calculate fop as the ratio of P32_open and P32
        '''
        if method == 'classic':
            if not {'p32'+str(flag), 'p32_open'+str(flag)}.issubset(self.analyses_dataframe_c.columns):
                raise Exception("You must compute open and full densities to compute fop")
            else:
                self.analyses_dataframe_c['fop'+str(flag)]=self.analyses_dataframe_c['p32_open'+str(flag)]/self.analyses_dataframe_c['p32'+str(flag)]
                return self.analyses_dataframe_c
        if method == 'smoothed':
            if not {'p32'+str(flag), 'p32_open'+str(flag)}.issubset(self.analyses_dataframe_s.columns):
                raise Exception("You must compute open and full densities to compute fop")
            else:
                self.analyses_dataframe_s['fop'+str(flag)]=self.analyses_dataframe_s['p32_open'+str(flag)]/self.analyses_dataframe_s['p32'+str(flag)]
                return self.analyses_dataframe_s

    def borehole_lithology_limits(self, lithology_type):
        '''put the lithology limits in a dataframe {'lithology':(up_lim,low_lim)}. 
        lithology_type: name of the lithology type. E.g. 'FRACTURE_DOMAIN'.
        '''
        litho_list = list()
        secup_list = list()
        seclow_list = list()
        elev_up_list = list()
        elev_low_list = list()
        litho1 = self._bdataframe[lithology_type].values[0]
        secup1 = self._bdataframe['ADJUSTEDSECUP(m)'].values[0]
        elev1 = self._bdataframe['ELEVATION_ADJUSTEDSECUP'].values[0]
        for index in self._bdataframe.index:
            litho2 = self._bdataframe[lithology_type][index]
            secup2 = self._bdataframe['ADJUSTEDSECUP(m)'][index]
            elev2 = self._bdataframe['ELEVATION_ADJUSTEDSECUP'][index]
            if litho2 != litho1:
                if litho1 not in [np.NaN, np.nan]:
                    litho_list.append(litho1)
                    secup_list.append(secup1)
                    elev_up_list.append(elev1)
                    seclow_list.append(secup2)
                    elev_low_list.append(elev2)
                litho1 = litho2
                secup1 = secup2
                elev1 = elev2
        if litho1 not in [np.NaN, np.nan]:
            litho_list.append(litho1)
            secup_list.append(secup1)
            seclow_list.append(secup2)
            elev_up_list.append(elev1)
            elev_low_list.append(elev2)

        self._all_litho_limits[lithology_type] = pd.DataFrame({'lithologies' : np.asarray(litho_list)})
        self._all_litho_limits[lithology_type]['secup'] = np.asarray(secup_list)
        self._all_litho_limits[lithology_type]['seclow'] = np.asarray(seclow_list) 
        self._all_litho_limits[lithology_type]['elev_up'] = np.asarray(elev_up_list) 
        self._all_litho_limits[lithology_type]['elev_low'] = np.asarray(elev_low_list) 
        self._all_litho_limits[lithology_type] = self._all_litho_limits[lithology_type].loc[self._all_litho_limits[lithology_type]['lithologies'].notna()]
        self._all_litho_limits[lithology_type]['sec_length'] = self._all_litho_limits[lithology_type]['seclow']-self._all_litho_limits[lithology_type]['secup']


    def plot_tadpole(self, ax, frac_type = 'all', **kwargs):
        '''
        '''
        x_label_size = kwargs.get('x_label_size', 15)
        if frac_type == 'all':
            dataframe_select = self._bdataframe.copy()
            color = 'b'
        elif frac_type == 'open':
            dataframe_select = self._bdataframe.loc[self._bdataframe['FRACT_INTERPRET'].isin(['Open','Partly open'])].copy()
            color = 'g'
        elif frac_type == 'sealed':
            dataframe_select = self._bdataframe.loc[self._bdataframe['FRACT_INTERPRET']=='Sealed'].copy()
            color = 'r'
        dipd = (dataframe_select['STRIKE(degrees)'] + 90) % 360
        dipd = dipd.values
        dip = dataframe_select['DIP(degrees)'].values
        depth = dataframe_select['ADJUSTEDSECUP(m)'].values
        ax.quiver(dip, depth, np.sin(dipd*math.pi/180), np.cos(dipd*math.pi/180), units='width', color = 'k', headwidth = 0, width = 0.005, scale = 10)
        ax.scatter(dip, depth, s=2, color = color)
        ax.set_xticks([0, 30, 60, 90])
        ax.set_xlabel('Dip '+str(frac_type), fontsize = x_label_size)
        ax.set_ylim(self._secup,self._seclow)
        ax.invert_yaxis()

    def plot_lithology(self, ax, lithology_type, **kwargs):
        '''
        To do: find a way to have lithology colormap global to all boreholes
        '''
        x_label = kwargs.get('x_label', lithology_type)
        x_label_size = kwargs.get('x_label_size', 15)
        legend = kwargs.get('legend', False)

        ax.set_xlabel(lithology_type, fontsize=x_label_size, rotation = 30, ha = 'right')

        if lithology_type not in self._all_litho_limits.keys():
            self.borehole_lithology_limits(lithology_type)                
            
        #---vector along borehole
        y = np.linspace(self._secup,self._seclow,int((self._seclow-self._secup))*5) 
        #---colors        
        b_lithology_list = list(self._all_litho_limits[lithology_type].groupby(['lithologies']).groups.keys())
        colormap = 'jet'
        cmap = plt.cm.get_cmap(colormap)
        colormap = cmap(np.linspace(0,1,len(b_lithology_list)))

        for index in self._all_litho_limits[lithology_type].index:
            litho = self._all_litho_limits[lithology_type]['lithologies'][index]
            secup = self._all_litho_limits[lithology_type]['secup'][index]
            seclow = self._all_litho_limits[lithology_type]['seclow'][index]
            color_index = b_lithology_list.index(litho)
            ax.fill_betweenx(y, 0, 1, where = np.logical_and(y>=secup, y<=seclow), facecolor = colormap[color_index], label = litho)
        ax.set_ylim(self._secup,self._seclow)
        ax.invert_yaxis()
        if legend == True:
            ax.legend()

    def plot(self, ax, x, y, display_type, **kwargs):
        '''
        '''
        _label = kwargs.get('label', None)
        #_color = kwargs.get('color', 'k')
        _xlabel = kwargs.get('xlabel', None)

        if _xlabel != None:
            ax.set_xlabel(_xlabel, fontsize = 15)
        ax.plot(x, y, display_type, label = _label)
        ax.set_ylim(self._secup,self._seclow)
        ax.invert_yaxis()
        if _label != None:
            ax.legend()




