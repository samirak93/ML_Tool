from bokeh.models.callbacks import CustomJS

callback_notification = CustomJS(args={}, code=
            """var x = document.getElementById("toast")
                x.className = "show";
                s = cb_obj.text
                document.getElementById("desc").innerHTML = s.substr(s.indexOf(' ')+1);
                setTimeout(function(){ x.className = x.className.replace("show", ""); }, 5000);""")


callback_button_plot = CustomJS(args={}, code=
            """$(document).ready(function() {
                $('.button').on('click', function() {
                var $this = $(this);
                var loadingText = '<p style="font-size:16px;text-align: center;">Plotting... <i class="fa fa-circle-o-notch fa-spin"></i></p>';
                console.log(loadingText)
                if ($(this).html() !== loadingText) {
                $this.data('original-text', $(this).html());
                $this.html(loadingText);}
                setTimeout(function() {
                $this.html($this.data('original-text'));}, 3000);
                });
                })""")