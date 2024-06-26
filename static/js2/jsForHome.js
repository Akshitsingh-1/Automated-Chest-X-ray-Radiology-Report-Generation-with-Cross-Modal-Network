$(document).ready(function () {
   // $("#start").hide();
    $(".loader").hide();
    var submitBtn = $('#start');
    submitBtn.attr('name', 'Get Started');
    // getStart
    $('#start').click(function () {
        //var form_data = new FormData($('#upload-file')[0]);
        
        // Show loading animation
       // $(this).hide();
        $('.loader').show();
        // $('html, body').animate({
        //     scrollTop: $("#generateReport").offset().top
        // }, 1000);

       // 
        // Move from homepage api /getStart
        // window.location.href = 'index.html';
        $.ajax({
            type: 'POST',
            url: '/getStart',
           // data: form_data,
            // contentType: false,
            // cache: false,
            // processData: false,
            // async: true,
            success: function () {
                setTimeout(function() {}, 3000);
                // $(document).hide();
                $(this).val('Start');
                $('.loader').hide();
                //window.location.href = '../../templates/index.html';
            },

        });
    });

});
