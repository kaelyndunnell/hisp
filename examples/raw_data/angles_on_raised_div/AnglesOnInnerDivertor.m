R_Top = 3.969525;
R_Bottom = 5.163925;
Z_Top = -2.555000;
Z_Bottom = -3.911605;

R_EOR = [
5.1609
5.285589
5.361321
5.515685
5.592035
5.505336
5.521551
5.508519
5.550099
5.356188
5.25381
5.199015
];

Z_EOR = [
-2.569555
-2.625938
-2.685585
-2.955929
-2.838186
-2.92992
-3.050399
-3.19555
-3.32592
-3.519511
-3.929313
-3.885961
];

Angle_EOR = [
1.898508
2.128559
2.135999
2.086305
2.133365
2.135598
2.551925
2.556929
2.910893
3.339399
3.939259
3.99229
];

R_interm = [
5.123131
5.235562
5.299562
5.352196
5.508393
5.555229
5.580923
5.505115
5.520161
5.521936
5.512185
5.588882
5.522292
5.351959
5.236296
5.199015
];

Z_interm = [
-2.559323
-2.599565
-2.63533
-2.699026
-2.936613
-2.989223
-2.859591
-2.925252
-3.018509
-3.065582
-3.159866
-3.250061
-3.381861
-3.556615
-3.96329
-3.885961
];

Angle_interm = [
2.558838
2.538305
2.555525
2.333025
2.029306
1.955001
1.986099
1.903966
1.660162
1.583525
1.596509
1.6002
1.915069
2.263119
2.68569
2.889992
];

R_EOD = [
5.1609
5.261283
5.335282
5.509255
5.559832
5.590321
5.515991
5.521651
5.518809
5.599395
5.553599
5.399838
5.29599
5.236296
5.188226
];

Z_EOD = [
-2.569555
-2.611289
-2.662389
-2.935193
-2.996555
-2.881969
-2.995309
-3.058182
-3.115315
-3.209259
-3.338508
-3.593023
-3.651281
-3.96329
-3.861999
];

Angle_EOD = [
2.815996
2.95636
2.35536
1.92055
1.5821
1.288599
1.0082
0.901512
0.925228
0.556261
0.916161
1.005998
1.29533
1.59621
1.636625
];

% polygonal line of divertor
Z = linspace(Z_Top, Z_Bottom);
REOR = ppval(spline(Z_EOR, R_EOR),Z);
AnglesEOR = ppval(spline(Z_EOR, Angle_EOR), Z);
Rinterm = ppval(spline(Z_interm, R_interm),Z);
Anglesinterm = ppval(spline(Z_interm, Angle_interm), Z);
REOD = ppval(spline(Z_EOD, R_EOD),Z);
AnglesEOD = ppval(spline(Z_EOD, Angle_EOD), Z);

% new basis based on the length on divertor surface
% recent version of MATLAB has 'arclength' function, but mine haven't it 
n = numel(Z);
XEOR = zeros(n, 1);
Xinterm = zeros(n, 1);
XEOD = zeros(n, 1);
for i = 1:n-1
  XEOR(i+1) = XEOR(i) + sqrt((Z(i+1)-Z(i))^2 + (REOR(i+1)-REOR(i))^2);
  Xinterm(i+1) = Xinterm(i) + sqrt((Z(i+1)-Z(i))^2 + (Rinterm(i+1)-Rinterm(i))^2); % simple Pythagoras
  XEOD(i+1) = XEOD(i) + sqrt((Z(i+1)-Z(i))^2 + (REOD(i+1)-REOD(i))^2);
end

tiledlayout(3,3)
% divertor geometry
nexttile; plot(Z_EOR, R_EOR,'o', Z, REOR,'r-'); title("EOR"); 
nexttile; plot(Z_interm, R_interm,'o', Z, Rinterm,'r-'); title("interm"); 
nexttile; plot(Z_EOD, R_EOD,'o', Z, REOD,'r-'); title("EOD"); 
% angle distribution on Z-axis (unit: Y=[degree], X=[m])
nexttile; plot(Z, AnglesEOR,'r-'); 
nexttile; plot(Z, Anglesinterm,'g-'); 
nexttile; plot(Z, AnglesEOD,'b-'); 
% angle trend from the bottom of inner divertor (unit: Y=[degree], X=[m])

nexttile; plot(XEOR, smooth(AnglesEOR,10),'r-'); hold on
            plot(Xinterm, smooth(Anglesinterm,10),'g-'); 
            plot(XEOD, smooth(AnglesEOD,10),'b-'); X=[0:0.05:1.699];
            title('angle')
            
            AnglesEOR=interp1(XEOR, AnglesEOR, X);
            Anglesinterm=interp1(Xinterm, Anglesinterm, X);
            AnglesEOD=interp1(XEOD, AnglesEOD, X);
            
            YEOR=1-sin((AnglesEOR)*pi/180)./sin((Anglesinterm)*pi/180);
            Yinterm=1-sin((Anglesinterm)*pi/180)./sin((Anglesinterm)*pi/180);
            YEOD=1-sin((AnglesEOD)*pi/180)./sin((Anglesinterm)*pi/180);
            
            YEOR=smooth(YEOR,10);
            Yinterm=smooth(Yinterm,10);
            YEOD=smooth(YEOD,10);
            
            for i=1:length(X)
                pf(i,:)=polyfit([0 1 2],[YEOR(i) Yinterm(i) YEOD(i)],1)
            end
%             pf(:,1)=(pf(:,1)-pf(:,2))/2;
%             pf(:,2)=-pf(:,1);
            
            for i=1:length(X)
                YEOR_fit(i)=0*pf(i,1)+pf(i,2);
                Yinterm_fit(i)=1*pf(i,1)+pf(i,2);
                YEOD_fit(i)=2*pf(i,1)+pf(i,2);
            end
                        
            plot(X, asin(sin(Anglesinterm*pi/180).*(1-YEOR_fit))*180/pi,'r:'); 
            plot(X, asin(sin(Anglesinterm*pi/180).*(1-Yinterm_fit))*180/pi,'g:'); 
            plot(X, asin(sin(Anglesinterm*pi/180).*(1-YEOD_fit))*180/pi,'b:'); hold off

nexttile; 
            plot(X, 1-YEOR,'r-'); hold on
            plot(X, 1-Yinterm,'g-');
            plot(X, 1-YEOD,'b-'); 
            title('sin(angle) relative to interm')
            
            plot(X, 1-YEOR_fit,'r:'); 
            plot(X, 1-Yinterm_fit,'g:'); 
            plot(X, 1-YEOD_fit,'b:'); hold off
            
nexttile;
            plot(X, pf(:,1),'c-'); hold on
            plot(X, pf(:,2),'m-'); hold off
            title('linear fit of relative sin(angle)')
            
            
            X=X';
            
            x_vals = [0.03588552908502555
                0.10050600153512998
                0.16100959902595532
                0.22095089328085088
                0.28109555910563105
                0.35619058685991935
                0.5112565155958036
                0.5915000805185952
                0.5308836928325285
                0.5905895358159668
                0.6509511006389596
                0.9158159281293858
                0.9918912083539122
                0.8850091396550829
                0.9852592135505992
                1.088580663953981
                1.2233599862333999
                1.3938109355006015
                1.5809020552652123
                ];
            
            coefficients_ = pf(:, 1);  % Slopes
            constants_ = pf(:, 2);     % Intercepts
            
            x_interpolated = interp1(X, X, x_vals, 'linear');
            
            coefficients = interp1(X, coefficients_, x_vals, 'linear');
            constants = interp1(X, constants_, x_vals, 'linear');

            y_vals = zeros(size(x_vals));
            
            for time = 0:160
                for i = 1:length(x_vals)
                    % Find the closest index in X to the interpolated x value
    %                 [~, idx] = min(abs(X - x_interpolated(i)));
                    % Use the slope and intercept from the corresponding segment
                    y_vals(i) = 1-polyval([coefficients(i) constants(i)], 2*time/160);
                end
                writematrix(y_vals,'angle_coeffs_t' + string(time) + '.txt', 'Delimiter',';');
            end
               

            % Display the interpolated y values
            disp('Interpolated y values:');
            disp(y_vals);            
            
            

            