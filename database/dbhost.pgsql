drop table staging.single_thread

select * from staging.thread_multi
limit 10;

select count(*) from staging.thread_multi;  --1729976

insert into staging.thread_multi
values (1186327459470, 'Efficacy-Facial Wash', '38,40', '紧绷_self', 'positive', 'positive');

select * from staging.thread_multi
offset 1729976;

select * from staging.skincare_review_nlp_airflow
limit 10;

drop table staging.skincare_review_nlp_airflow;
create table staging.skincare_review_nlp_airflow as 
select * from skincare_review_nlp_form_dedup
limit 10;

select * from staging.skincare_review_raw
limit 100;