require 'test_helper'

class HangOutsControllerTest < ActionDispatch::IntegrationTest
  setup do
    @hang_out = hang_outs(:one)
  end

  test "should get index" do
    get hang_outs_url
    assert_response :success
  end

  test "should get new" do
    get new_hang_out_url
    assert_response :success
  end

  test "should create hang_out" do
    assert_difference('HangOut.count') do
      post hang_outs_url, params: { hang_out: { date: @hang_out.date, end_time: @hang_out.end_time, name: @hang_out.name, start_time: @hang_out.start_time, user_id: @hang_out.user_id } }
    end

    assert_redirected_to hang_out_url(HangOut.last)
  end

  test "should show hang_out" do
    get hang_out_url(@hang_out)
    assert_response :success
  end

  test "should get edit" do
    get edit_hang_out_url(@hang_out)
    assert_response :success
  end

  test "should update hang_out" do
    patch hang_out_url(@hang_out), params: { hang_out: { date: @hang_out.date, end_time: @hang_out.end_time, name: @hang_out.name, start_time: @hang_out.start_time, user_id: @hang_out.user_id } }
    assert_redirected_to hang_out_url(@hang_out)
  end

  test "should destroy hang_out" do
    assert_difference('HangOut.count', -1) do
      delete hang_out_url(@hang_out)
    end

    assert_redirected_to hang_outs_url
  end
end
