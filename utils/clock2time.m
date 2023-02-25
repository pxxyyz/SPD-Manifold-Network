function filetime = clock2time(str)
filetime=[num2str(str(1)),'-',num2str(str(2)),'-',num2str(str(3)),...
    '-',num2str(str(4)),'-',num2str(str(5)),'-',num2str(round(str(6))) ];
end