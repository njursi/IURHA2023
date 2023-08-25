%% 恒虚警检测+STFT,采用外部函数
        aZ = abs(Z).^2;
        N_fft = 128;  % 太大会模糊，建议128
        X = zeros(Nscn,N_fft);
        N=50;
        pro_N=6;
        PAD=10^(-4);

        for n = N_fft/2:Nscn-N_fft/2
            [x_detected, XT ] = cfar_ac(aZ(n,:), N, pro_N, PAD);
            if isempty(x_detected) == 0
                for j = 1:length(x_detected)
                    x = Z(n-N_fft/2+1:n+N_fft/2,x_detected(j))';
                    Fx = fftshift(abs(fft(x,N_fft)));
                    X(n,:) = Fx + X(n,:);
                end
            end
        end
        X = X(N_fft/2:Nscn-N_fft/2,:);
        %figure('Units','normalized','Position',[0.3 0.3 0.4 0.5],'Color','w','MenuBar','figure')
        nexttile;
        X = 255*(X-min(min(X))) / (max(max(X))-min(min(X))); % 归一化
        imagesc(tscn,Vfft,X')
        xlim([0,5]);ylim([-2,2]);
        colormap(gcf,'jet');
        title(strcat('Person:',num2str(pe),{32},'Class:',num2str(m)), ...
              'FontSize',12,'FontName','Times New Roman','FontWeight','bold')
        show_i = show_i + 1;



%% 计算全局的速度-多普勒图，采用加权距离时频变换
        Ts = mean(diff(tscn));
        X_sum2 = 0;
        Ndft = 256;
        w_l = fix(Fscn);
        ratio = 0.995;
        aZ = abs(Z);
        E = sum(sum(aZ));
        for i = 1:Nbin
            w_i = sum(aZ(:,i))/E;
            if i<200; w_i = 0; end %去除耦合波；
            % 计算每个距离bin下的STFT时频图
            X2 = stft(Z(:,i),Ts,'Window',hann(w_l,'periodic') ,'OverlapLength',fix(w_l*ratio),'FFTLength',Ndft);
            X_sum2 = X_sum2 + w_i * abs(X2);
        end

        %figure('Units','normalized','Position',[0.3 0.3 0.4 0.5],'Color','w','MenuBar','figure')
        nexttile;
        X_sum2 = 255*(X_sum2-min(min(X_sum2))) / (max(max(X_sum2))-min(min(X_sum2))); % 归一化
        imagesc(tscn,Vfft,flipud(X_sum2))
        xlim([0,5]);ylim([-2,2]);
        title(strcat('Person:',num2str(pe),{32},'Class:',num2str(m)), ...
            'FontSize',12,'FontName','Times New Roman','FontWeight','bold')
        colormap(gcf,'jet');
        show_i = show_i + 1;



%% 计算三维的多普勒-时间-距离矩阵并作二维切割
        Ts = mean(diff(tscn));
        DTR_3D = [];
        Ndft = 256;
        w_l = fix(Fscn/2);
        ratio = 0.995;
        for i = 1:Nbin
            % 计算每个距离bin下的STFT时频图
            X2 = stft(Z(:,i),Ts,'Window',hann(w_l,'periodic') ,'OverlapLength',fix(w_l*ratio),'FFTLength',Ndft);
            DTR_3D(:,:,i) = abs(X2);
        end
        number_show = fix(size(DTR_3D,2)/2);
        DTR_3D_slice = squeeze(DTR_3D(:,number_show,200:576));
        DTR_3D_slice = 255*(DTR_3D_slice-min(min(DTR_3D_slice))) / (max(max(DTR_3D_slice))-min(min(DTR_3D_slice))); % 归一化
        nexttile;
        imagesc(R(200:576),Vfft,flipud(DTR_3D_slice));
        ylim([-2,2]);
        title(strcat('Person:',num2str(pe),{32},'Class:',num2str(m)), ...
              'FontSize',12,'FontName','Times New Roman','FontWeight','bold')  % {32}表示空格
        show_i = show_i + 1;

%% 滤波器参数设置
b = [0.0103 0.0619 0.1547 0.2063 0.1547 0.0619 0.0103];
a = [1.0000 -1.1876 1.3052 -0.6743 0.2635 -0.0518 0.0050];
edat = max(filter(b,a,abs(X_MTI),[],2),0);
