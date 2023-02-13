import mlreportgen.report.*
import mlreportgen.dom.*
rpt = Report('peaks');

surf(peaks(20));
figure = Figure();
peaks20 = Image(getSnapshotImage(figure,rpt));
peaks20.Width = '3in';
peaks20.Height = [];
delete(gcf);

surf(peaks(40));
figure = Figure();
peaks40 = Image(getSnapshotImage(figure,rpt));
peaks40.Width = '3in';
peaks40.Height = [];
delete(gcf);

t = Table({peaks20,peaks40;'peaks(20)','peaks(40)'});
add(rpt,t);
close(rpt);
rptview(rpt);