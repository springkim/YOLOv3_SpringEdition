using System;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;


namespace YOLOv3_CS_Example
{
    public partial class Form1 : Form
    {
        [DllImport("libYOLOv3SE.dll", EntryPoint = "YoloLoad", CallingConvention = CallingConvention.Cdecl)]
        static extern IntPtr YoloLoad(String cfg,String weights);
        [DllImport("libYOLOv3SE.dll", EntryPoint = "YoloDetectFromFile", CallingConvention = CallingConvention.Cdecl)]
        static extern int YoloDetectFromFile(String img_path,IntPtr net,float threshold,float[] result,int result_sz);
        [DllImport("libYOLOv3SE.dll", EntryPoint = "YoloRelease", CallingConvention = CallingConvention.Cdecl)]
        static extern int YoloRelease(IntPtr net);

        String image_file;
        IntPtr net = IntPtr.Zero;
        public Form1()
        {
            InitializeComponent();
            pictureBox1.Anchor = (AnchorStyles.Bottom | AnchorStyles.Right | AnchorStyles.Top | AnchorStyles.Left);
            btn_load.Anchor = (AnchorStyles.Bottom);
            btn_detect.Anchor = (AnchorStyles.Bottom);
            btn_openimage.Anchor = (AnchorStyles.Bottom);
            btn_release.Anchor = (AnchorStyles.Bottom);
        }

        private void OnClickLoad(object sender, EventArgs e)
        {
            if (net != IntPtr.Zero)
            {
                YoloRelease(net);
            }
            String weights="", cfg="";
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Weights Files(*.weights)|*.weights|All files (*.*)|*.*";
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                weights = ofd.FileName;
                ofd.Filter = "Cfg Files(*.cfg)|*.cfg|All files (*.*)|*.*";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    cfg = ofd.FileName;
                }
            }
            if (cfg != "" && weights != "")
            {
                this.net=YoloLoad(cfg, weights);
            }else
            {
                this.net = IntPtr.Zero;
            }
        }

        private void OnClickDetect(object sender, EventArgs e)
        {
            if (this.net != IntPtr.Zero)
            {
                float[] result = new float[1024];
                int n=YoloDetectFromFile(image_file, net, 0.5F, result, 1024);
                FileStream fs = new FileStream(image_file, FileMode.Open, FileAccess.Read);
                Image img = System.Drawing.Image.FromStream(fs);
                fs.Close();
                Graphics g = Graphics.FromImage(img);
                Pen pen = new Pen(Color.Red, 5);

                for (int i = 0; i < n; i++)
                {
                    int c= (int)result[i * 6 + 0];
                    int x = (int)result[i * 6 + 2];
                    int y = (int)result[i * 6 + 3];
                    int w = (int)result[i * 6 + 4];
                    int h = (int)result[i * 6 + 5];
                    Rectangle rect = new Rectangle(x, y, w, h);
                    g.DrawRectangle(pen, rect);
                }
                pictureBox1.Image = img;
            }
        }

        private void OnClickRelease(object sender, EventArgs e)
        {
            if (net != IntPtr.Zero)
            {
                YoloRelease(net);
            }
        }

        private void OnClickOpenImage(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Image Files(*.BMP;*.JPG;*.GIF)|*.BMP;*.JPG;*.GIF|All files (*.*)|*.*";


            if (ofd.ShowDialog() == DialogResult.OK)
            {

                FileStream fs = new FileStream(ofd.FileName, FileMode.Open, FileAccess.Read);
                Image img = System.Drawing.Image.FromStream(fs);
                fs.Close();

                pictureBox1.Image = img;
                pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;
                this.image_file = ofd.FileName;
            }
        }
    }
}
