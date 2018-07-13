namespace YOLOv3_CS_Example
{
    partial class Form1
    {
        /// <summary>
        /// 필수 디자이너 변수입니다.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 사용 중인 모든 리소스를 정리합니다.
        /// </summary>
        /// <param name="disposing">관리되는 리소스를 삭제해야 하면 true이고, 그렇지 않으면 false입니다.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form 디자이너에서 생성한 코드

        /// <summary>
        /// 디자이너 지원에 필요한 메서드입니다. 
        /// 이 메서드의 내용을 코드 편집기로 수정하지 마세요.
        /// </summary>
        private void InitializeComponent()
        {
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.btn_load = new System.Windows.Forms.Button();
            this.btn_detect = new System.Windows.Forms.Button();
            this.btn_release = new System.Windows.Forms.Button();
            this.btn_openimage = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.SuspendLayout();
            // 
            // pictureBox1
            // 
            this.pictureBox1.Location = new System.Drawing.Point(12, 12);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(488, 339);
            this.pictureBox1.TabIndex = 0;
            this.pictureBox1.TabStop = false;
            // 
            // btn_load
            // 
            this.btn_load.Location = new System.Drawing.Point(12, 382);
            this.btn_load.Name = "btn_load";
            this.btn_load.Size = new System.Drawing.Size(75, 23);
            this.btn_load.TabIndex = 1;
            this.btn_load.Text = "Load";
            this.btn_load.UseVisualStyleBackColor = true;
            this.btn_load.Click += new System.EventHandler(this.OnClickLoad);
            // 
            // btn_detect
            // 
            this.btn_detect.Location = new System.Drawing.Point(290, 382);
            this.btn_detect.Name = "btn_detect";
            this.btn_detect.Size = new System.Drawing.Size(75, 23);
            this.btn_detect.TabIndex = 2;
            this.btn_detect.Text = "Detect";
            this.btn_detect.UseVisualStyleBackColor = true;
            this.btn_detect.Click += new System.EventHandler(this.OnClickDetect);
            // 
            // btn_release
            // 
            this.btn_release.Location = new System.Drawing.Point(425, 381);
            this.btn_release.Name = "btn_release";
            this.btn_release.Size = new System.Drawing.Size(75, 23);
            this.btn_release.TabIndex = 3;
            this.btn_release.Text = "Release";
            this.btn_release.UseVisualStyleBackColor = true;
            this.btn_release.Click += new System.EventHandler(this.OnClickRelease);
            // 
            // btn_openimage
            // 
            this.btn_openimage.Location = new System.Drawing.Point(131, 380);
            this.btn_openimage.Name = "btn_openimage";
            this.btn_openimage.Size = new System.Drawing.Size(93, 23);
            this.btn_openimage.TabIndex = 4;
            this.btn_openimage.Text = "OpenImage";
            this.btn_openimage.UseVisualStyleBackColor = true;
            this.btn_openimage.Click += new System.EventHandler(this.OnClickOpenImage);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(512, 439);
            this.Controls.Add(this.btn_openimage);
            this.Controls.Add(this.btn_release);
            this.Controls.Add(this.btn_detect);
            this.Controls.Add(this.btn_load);
            this.Controls.Add(this.pictureBox1);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Button btn_load;
        private System.Windows.Forms.Button btn_detect;
        private System.Windows.Forms.Button btn_release;
        private System.Windows.Forms.Button btn_openimage;
    }
}

